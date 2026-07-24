"""Phase D — skills earned from repetition, and the correlation of their worth.

The family covenant is the door; the tests knock on it from every side.
"""

import json

import pytest

from agent import chronicle, earned_skills, soul


@pytest.fixture(autouse=True)
def soul_on(monkeypatch):
    monkeypatch.setenv("HERMES_SOUL", "1")


def _complete_episode(turn_id):
    episode = soul.begin_episode(turn_id, task_id="t", session_id="s")
    episode.record(soul.TOOL_CALL, tool="write_file", arguments="{}")
    episode.seal(soul.OUTCOME_COMPLETE)
    return episode.episode_id


def _failed_episode(turn_id):
    episode = soul.begin_episode(turn_id)
    episode.seal(soul.OUTCOME_FAILED)
    return episode.episode_id


def _lesson(lid, episode_id, *, tags=("files",), outcome=soul.OUTCOME_COMPLETE):
    lesson = {"id": lid, "title": lid, "text": f"when {lid}: x",
              "tags": list(tags), "keywords": [],
              "source_episode": episode_id, "source_outcome": outcome}
    assert chronicle.add(lesson)
    return lesson


def _family_of(n, tag="files"):
    """n lessons from n distinct verified-complete episodes, sharing a tag."""
    for i in range(n):
        _lesson(f"L{i}", _complete_episode(f"turn-{i}"), tags=(tag,))


GOOD_SKILL = {"title": "Quote paths with spaces",
              "reach_for_when": "writing to paths that may contain spaces",
              "steps": ["quote the path", "verify the write landed"],
              "tags": ["files"], "keywords": ["path", "quote"]}


def _draft(*episodes, skill=None, tag="files"):
    return {"family_tag": tag, "episodes": list(episodes),
            "skill": dict(GOOD_SKILL) if skill is None else skill}


# --- the family covenant: the only door ----------------------------------


def test_three_verified_complete_episodes_make_a_family():
    _family_of(3)
    families = earned_skills.gather_families()
    assert len(families) == 1
    assert families[0]["tag"] == "files"
    assert len(families[0]["episodes"]) == 3


def test_two_episodes_are_not_yet_a_family():
    _family_of(2)
    assert earned_skills.gather_families() == []


def test_failed_episodes_never_count_toward_a_family():
    # Failed toil teaches lessons; only what repeatedly WORKED may become a
    # playbook.
    _family_of(2)
    _lesson("L-failed", _failed_episode("turn-f"), outcome=soul.OUTCOME_FAILED)
    assert earned_skills.gather_families() == []


def test_an_episode_that_aged_out_of_the_soul_lends_no_count():
    # Strictness over convenience: the lesson claims complete, but the Soul
    # can no longer verify it.
    _family_of(2)
    _lesson("L-gone", "episode-that-aged-out", outcome=soul.OUTCOME_COMPLETE)
    assert earned_skills.gather_families() == []


def test_a_consolidated_lesson_lends_words_but_no_episode():
    _family_of(2)
    consolidated = {"id": "consolidated-t-1", "title": "merged", "text": "when x: y",
                    "tags": ["files"], "keywords": [], "source_episode": None,
                    "source_outcome": None}
    chronicle.add(consolidated)
    families = earned_skills.gather_families(min_family=2)
    assert len(families[0]["lessons"]) == 3   # its words are present
    assert len(families[0]["episodes"]) == 2  # its count is not


def test_two_lessons_from_one_episode_count_once():
    episode_id = _complete_episode("turn-0")
    _lesson("L0", episode_id)
    _lesson("L1", episode_id)
    _lesson("L2", _complete_episode("turn-1"))
    assert earned_skills.gather_families() == []  # 2 distinct episodes, not 3


def test_a_tag_already_worn_by_a_skill_is_not_re_dreamt():
    _family_of(3)
    earned_skills.add({"id": "skill-x", "title": "existing", "text": "t",
                       "tags": ["files"], "keywords": []})
    assert earned_skills.gather_families() == []


# --- the dream: propose, never apply -------------------------------------


def test_the_dream_drafts_from_a_ready_family_and_shelves_nothing():
    _family_of(3)
    got = earned_skills.propose_skills(
        lambda prompt: json.dumps({"skill": GOOD_SKILL}))
    assert len(got["proposal"]["drafts"]) == 1
    assert got["proposal"]["drafts"][0]["episodes"]
    assert earned_skills.entries() == []  # nothing shelved


def test_an_honest_empty_hand_yields_no_draft():
    _family_of(3)
    got = earned_skills.propose_skills(lambda prompt: json.dumps({"skill": None}))
    assert got["proposal"]["drafts"] == []


def test_skip_tags_leaves_a_waiting_family_alone():
    _family_of(3, tag="files")
    got = earned_skills.propose_skills(
        lambda prompt: json.dumps({"skill": GOOD_SKILL}),
        skip_tags=["files"], write=False)
    assert got["proposal"]["drafts"] == []


# --- apply: the covenant is law ------------------------------------------


def test_a_blessed_draft_is_shelved_with_its_provenance():
    got = earned_skills.apply_skills(
        {"proposed_at": "t1", "drafts": [_draft("e1", "e2", "e3")]})
    assert len(got["shelved"]) == 1
    skill = earned_skills.entries()[0]
    assert skill["id"] == "skill-t1-1"
    assert skill["from_episodes"] == ["e1", "e2", "e3"]
    assert "Reach for this when:" in skill["text"]


def test_provenance_below_the_family_floor_is_refused():
    got = earned_skills.apply_skills(
        {"proposed_at": "t1", "drafts": [_draft("e1", "e2")]})
    assert got["shelved"] == []
    assert "provenance below 3" in got["refused"][0]["why"]


def test_a_card_too_long_is_refused_not_trimmed():
    bloated = dict(GOOD_SKILL, steps=["x" * 300 for _ in range(6)])
    got = earned_skills.apply_skills(
        {"proposed_at": "t1", "drafts": [_draft("e1", "e2", "e3", skill=bloated)]})
    assert got["shelved"] == []
    assert "a card, not a book" in got["refused"][0]["why"]


@pytest.mark.parametrize("broken", [
    dict(GOOD_SKILL, title=""),
    dict(GOOD_SKILL, reach_for_when=""),
    dict(GOOD_SKILL, steps=[]),
    dict(GOOD_SKILL, steps="not a list"),
])
def test_a_draft_missing_its_parts_is_refused(broken):
    got = earned_skills.apply_skills(
        {"proposed_at": "t1", "drafts": [_draft("e1", "e2", "e3", skill=broken)]})
    assert got["shelved"] == []
    assert got["refused"]


def test_the_family_tag_keeps_kinship_findable():
    draft = _draft("e1", "e2", "e3", skill=dict(GOOD_SKILL, tags=["other"]))
    earned_skills.apply_skills({"proposed_at": "t1", "drafts": [draft]})
    assert "files" in earned_skills.entries()[0]["tags"]


def test_ids_keep_their_position_so_partial_blessing_never_collides():
    proposal = {"proposed_at": "t1",
                "drafts": [_draft("e1", "e2", "e3"),
                           _draft("e4", "e5", "e6", tag="net",
                                  skill=dict(GOOD_SKILL, title="net skill"))]}
    earned_skills.apply_skills(proposal, indices=[1])  # bless draft 2 first
    earned_skills.apply_skills(proposal, indices=[0])  # then draft 1
    assert sorted(s["id"] for s in earned_skills.entries()) == \
        ["skill-t1-1", "skill-t1-2"]


# --- the desk: per-draft resolution --------------------------------------


def _proposal_on_desk(*drafts):
    got = earned_skills.propose_skills(
        lambda prompt: json.dumps({"skill": GOOD_SKILL}), write=False)
    proposal = {"proposed_at": soul.now_ts(), "family_count": len(drafts),
                "drafts": list(drafts)}
    earned_skills.proposals_dir().mkdir(parents=True, exist_ok=True)
    path = earned_skills.proposals_dir() / f"proposal-{proposal['proposed_at']}.json"
    path.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    return proposal["proposed_at"]


def test_shelving_a_draft_goes_through_the_full_covenant():
    ts = _proposal_on_desk(_draft("e1", "e2", "e3"))
    got = earned_skills.resolve_draft(ts, 0, "shelved")
    assert got["resolution"] == "shelved"
    assert len(earned_skills.entries()) == 1
    assert earned_skills.pending_drafts() == []


def test_a_covenant_refusal_leaves_the_draft_pending():
    ts = _proposal_on_desk(_draft("e1", "e2"))  # provenance too thin
    got = earned_skills.resolve_draft(ts, 0, "shelved")
    assert got["resolution"] is None
    assert got["refused"]
    assert len(earned_skills.pending_drafts()) == 1  # still awaiting a hand


def test_set_aside_marks_without_shelving():
    ts = _proposal_on_desk(_draft("e1", "e2", "e3"))
    got = earned_skills.resolve_draft(ts, 0, "set_aside")
    assert got["resolution"] == "set_aside"
    assert earned_skills.entries() == []
    assert earned_skills.pending_drafts() == []


def test_a_resolved_draft_takes_no_second_word():
    ts = _proposal_on_desk(_draft("e1", "e2", "e3"))
    earned_skills.resolve_draft(ts, 0, "set_aside")
    with pytest.raises(ValueError, match="already set_aside"):
        earned_skills.resolve_draft(ts, 0, "shelved")


def test_resolving_what_does_not_exist_raises():
    with pytest.raises(KeyError):
        earned_skills.resolve_draft("no-such-ts", 0, "shelved")
    ts = _proposal_on_desk(_draft("e1", "e2", "e3"))
    with pytest.raises(KeyError):
        earned_skills.resolve_draft(ts, 9, "shelved")


# --- serving -------------------------------------------------------------


def _shelve_skill(sid="skill-1", tags=("files",)):
    skill = {"id": sid, "title": sid, "text": "Reach for this when: x\nSteps:\n1. y",
             "tags": list(tags), "keywords": []}
    earned_skills.add(skill)
    return skill


def test_a_matching_playbook_is_served_and_an_irrelevant_one_is_not():
    _shelve_skill(tags=("files",))
    assert len(earned_skills.retrieve("rename these files")) == 1
    assert earned_skills.retrieve("write a poem about the sea") == []


def test_at_most_two_playbooks_ride_beside_one_task():
    for i in range(5):
        _shelve_skill(f"skill-{i}", tags=("files",))
    assert len(earned_skills.retrieve("files")) == earned_skills.SERVE_K_SKILLS


def test_served_skills_are_recorded_beside_the_outcome():
    soul.record_turn(turn_id="turn-z", messages=[], final_response="done",
                     completed=True, served_lessons=["L1"],
                     served_skills=["skill-1"])
    events = soul.read_episode(soul.list_episodes()[-1])
    assemble = next(e for e in events if e["type"] == soul.ASSEMBLE)
    assert assemble["lessons"] == ["L1"]
    assert assemble["skills"] == ["skill-1"]


# --- correlation: usage is not effectiveness -----------------------------


def _served_turn(skill_id, outcome, turn_id):
    soul.record_turn(turn_id=turn_id, messages=[], final_response="x",
                     completed=(outcome == soul.OUTCOME_COMPLETE),
                     failed=(outcome == soul.OUTCOME_FAILED),
                     served_skills=[skill_id])


def test_the_report_ties_serving_to_sealed_outcomes():
    _shelve_skill("skill-good")
    _served_turn("skill-good", soul.OUTCOME_COMPLETE, "t1")
    _served_turn("skill-good", soul.OUTCOME_COMPLETE, "t2")
    _served_turn("skill-good", soul.OUTCOME_FAILED, "t3")

    row = earned_skills.correlation_report()[0]
    assert row["served"] == 3
    assert row["complete"] == 2
    assert row["rate"] == pytest.approx(2 / 3)


def test_served_and_struggling_surfaces_before_served_and_thriving():
    _shelve_skill("skill-good")
    _shelve_skill("skill-bad")
    _served_turn("skill-good", soul.OUTCOME_COMPLETE, "t1")
    _served_turn("skill-bad", soul.OUTCOME_FAILED, "t2")

    rows = earned_skills.correlation_report()
    assert [r["id"] for r in rows] == ["skill-bad", "skill-good"]


def test_a_never_served_skill_reports_no_data_not_zero():
    _shelve_skill("skill-unserved")
    _shelve_skill("skill-bad")
    _served_turn("skill-bad", soul.OUTCOME_FAILED, "t1")

    rows = earned_skills.correlation_report()
    # No evidence must never outrank known struggle.
    assert rows[0]["id"] == "skill-bad"
    assert rows[1]["rate"] is None
    assert rows[1]["served"] == 0
