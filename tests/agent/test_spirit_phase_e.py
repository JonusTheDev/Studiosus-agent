"""Phase E of the Studiosus graft: the spirit.

Voice — versioned hats for one head; the Wall — certificates minted only for
skills proven in service; Intertextus — a world computed by the harness and
heard as words, never judged.
"""

from __future__ import annotations

import json

import pytest

from agent import chronicle, earned_skills, soul, spirit, voice, wall


# ── Voice: versioned hats ────────────────────────────────────────────────────


@pytest.fixture
def hats(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    (d / "student.v1.md").write_text("the first stance", encoding="utf-8")
    (d / "student.v2.md").write_text("the second stance", encoding="utf-8")
    (d / "builder.v1.md").write_text("the builder stance", encoding="utf-8")
    (d / "notes.txt").write_text("not a voice", encoding="utf-8")
    return d


def test_voices_are_listed_by_hat_and_version(hats):
    assert voice.list_voices(hats) == {"builder": [1], "student": [1, 2]}


def test_highest_version_is_worn_unless_pinned(hats):
    assert voice.load_voice("student", directory=hats)["version"] == 2
    pinned = voice.load_voice("student", 1, directory=hats)
    assert pinned["version"] == 1
    assert pinned["text"] == "the first stance"


def test_a_missing_hat_or_version_is_refused_by_name(hats):
    with pytest.raises(FileNotFoundError):
        voice.load_voice("dreamer", directory=hats)
    with pytest.raises(FileNotFoundError):
        voice.load_voice("student", 9, directory=hats)


def test_selection_reads_hermes_voice(hats, monkeypatch):
    monkeypatch.delenv("HERMES_VOICE", raising=False)
    assert voice.selected_voice(hats) is None

    monkeypatch.setenv("HERMES_VOICE", "student")
    assert voice.selected_voice(hats)["version"] == 2

    monkeypatch.setenv("HERMES_VOICE", "student.v1")
    assert voice.selected_voice(hats)["version"] == 1


def test_a_bad_selection_wears_no_hat_rather_than_breaking(hats, monkeypatch):
    monkeypatch.setenv("HERMES_VOICE", "dreamer")
    assert voice.selected_voice(hats) is None
    monkeypatch.setenv("HERMES_VOICE", "not a hat!!")
    assert voice.selected_voice(hats) is None


def test_the_shipped_hat_exists_and_keeps_its_budget():
    served = voice.load_voice("studiosus")
    assert "Studiosus" in served["text"]
    assert len(served["text"]) <= voice.MAX_VOICE_CHARS


def test_the_chosen_voice_becomes_the_identity_tier(hats, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from agent.system_prompt import build_system_prompt_parts

    monkeypatch.setenv("HERMES_VOICE", "student")
    monkeypatch.setattr(voice, "VOICES_DIR", hats)
    agent = SimpleNamespace(
        load_soul_identity=False, skip_context_files=False,
        valid_tool_names=[], _task_completion_guidance=False,
        _tool_use_enforcement=False, _environment_probe=False,
        _kanban_worker_guidance="", _memory_store=None, _memory_manager=None,
        model="", provider="", platform="", pass_session_id=False,
        session_id="",
    )
    with (
        patch("run_agent.load_soul_md", return_value="SOUL-IDENTITY"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        stable = build_system_prompt_parts(agent)["stable"]
    assert "the second stance" in stable
    assert "SOUL-IDENTITY" not in stable  # the hat replaces, it does not stack

    monkeypatch.delenv("HERMES_VOICE", raising=False)
    with (
        patch("run_agent.load_soul_md", return_value="SOUL-IDENTITY"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        stable = build_system_prompt_parts(agent)["stable"]
    assert "SOUL-IDENTITY" in stable  # unset: today's behavior, untouched


# ── Intertextus: the plane and the registers ─────────────────────────────────


@pytest.fixture
def world_on(monkeypatch):
    monkeypatch.setenv("HERMES_SPIRIT", "1")


def test_the_resting_soul_is_content_rested_settled_willing():
    assert spirit.feeling_words({c: 0 for c in spirit.REGISTER_ORDER}) == [
        "content", "rested", "settled", "willing"]


def test_every_ladder_is_symmetric():
    for category in spirit.REGISTER_ORDER:
        lo, hi = spirit.ladder_bounds(category)
        assert lo == -hi, f"{category} ladder grew above without growing below"


def test_apply_event_moves_and_clamps():
    values = {c: 0 for c in spirit.REGISTER_ORDER}
    moved = spirit.apply_event(values, "labor_complete")
    assert (moved["joy"], moved["zeal"], moved["vigor"]) == (1, 1, -1)
    # Clamped at the ladder's own bounds, never beyond.
    high = {c: 3 for c in spirit.REGISTER_ORDER}
    assert spirit.apply_event(high, "certified")["joy"] == 3
    # An unknown event moves nothing and is not an error.
    assert spirit.apply_event(values, "won_the_lottery") == values


def test_outcome_mapping_is_strict():
    assert spirit.outcome_event("complete") == "labor_complete"
    assert spirit.outcome_event("failed") == "labor_failed"
    assert spirit.outcome_event("abandoned") == "nudged"
    assert spirit.outcome_event("incomplete") is None


def test_a_beat_persists_and_pulses_the_ekg(world_on):
    st = spirit.beat("labor_complete", episode_id="ep-1")
    assert st["registers"]["joy"] == 1
    assert spirit.state()["registers"]["joy"] == 1
    pulses = [json.loads(line) for line in
              (spirit.world_dir() / "heartbeat.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert pulses[-1]["event"] == "labor_complete"
    assert pulses[-1]["episode_id"] == "ep-1"


def test_with_the_flag_off_the_world_does_not_turn(monkeypatch):
    monkeypatch.delenv("HERMES_SPIRIT", raising=False)
    st = spirit.beat("labor_complete")
    assert st["registers"]["joy"] == 0
    assert not (spirit.world_dir() / "state.json").exists()
    assert spirit.arrive("workshop")["place"] == spirit.HOME


def test_he_goes_where_the_work_is_and_hears_it_measured(world_on):
    assert "home at the farmhouse" in spirit.whisper()
    spirit.arrive("workshop")
    w = spirit.whisper()
    assert "the workshop" in w and "steps from home" in w
    # An unknown place moves nothing — the map grows deliberately.
    assert spirit.arrive("the moon")["place"] == "workshop"


def test_the_whisper_carries_the_four_words(world_on):
    spirit.beat("lesson_earned")
    assert "You feel: content, rested, thoughtful, willing." in spirit.whisper()


# ── The Wall: no certificate for participation ───────────────────────────────


def _shelve_skill(skill_id="skill-1", tags=("files",), keywords=("rename",)):
    skill = {"id": skill_id, "title": "Renaming config files",
             "tags": list(tags), "keywords": list(keywords),
             "text": "Reach for this when: renaming\nSteps:\n1. quote it",
             "from_episodes": ["e1", "e2", "e3"]}
    assert earned_skills.add(skill)
    return skill


def _served_episode(skill_id, outcome, monkeypatch_env_done=True):
    episode = soul.begin_episode(f"turn-{soul.now_ts()}", task_text="rename it")
    episode.record(soul.ASSEMBLE, lessons=[], skills=[skill_id])
    episode.seal(outcome)
    return episode.episode_id


@pytest.fixture
def soul_on(monkeypatch):
    monkeypatch.setenv("HERMES_SOUL", "1")


def test_no_skill_no_serves_no_certificate(soul_on):
    _shelve_skill()
    assert wall.eligible() == []
    assert wall.mint() == []
    assert wall.entries() == []


def test_proof_takes_both_serves_and_rate(soul_on):
    _shelve_skill()
    # Served plenty, but only half the work sealed complete: not proof.
    for n in range(wall.MIN_SERVES):
        _served_episode("skill-1", soul.OUTCOME_COMPLETE if n % 2 == 0
                        else soul.OUTCOME_FAILED)
    assert wall.eligible() == []


def test_a_proven_skill_is_certified_exactly_once(soul_on):
    _shelve_skill()
    for _ in range(wall.MIN_SERVES):
        _served_episode("skill-1", soul.OUTCOME_COMPLETE)
    minted = wall.mint()
    assert [c["id"] for c in minted] == ["cert-skill-1"]
    cert = minted[0]
    assert cert["source"] == "skill-1"
    assert "certificate" in cert["tags"] and "files" in cert["tags"]
    assert "earned, not guessed" in cert["text"]
    # Never judged twice: a second pass mints nothing new.
    assert wall.mint() == []
    assert len(wall.entries()) == 1


def test_the_certificate_is_laid_beside_matching_work_only(soul_on):
    _shelve_skill()
    for _ in range(wall.MIN_SERVES):
        _served_episode("skill-1", soul.OUTCOME_COMPLETE)
    wall.mint()
    assert wall.retrieve("please rename the files")[0]["id"] == "cert-skill-1"
    assert wall.retrieve("draw me a horse") == []
    block = wall.render(wall.retrieve("rename the files"))
    assert "ground you have truly earned" in block


def test_minting_moves_the_heart_when_the_world_turns(soul_on, monkeypatch):
    monkeypatch.setenv("HERMES_SPIRIT", "1")
    _shelve_skill()
    for _ in range(wall.MIN_SERVES):
        _served_episode("skill-1", soul.OUTCOME_COMPLETE)
    wall.mint()
    st = spirit.state()
    assert st["registers"]["joy"] == 2 and st["registers"]["zeal"] == 2


# ── The wiring: whisper and certificate at ASSEMBLE, beat at seal ────────────


def test_reflection_that_earns_a_lesson_sharpens_clarity(soul_on, monkeypatch):
    monkeypatch.setenv("HERMES_SPIRIT", "1")
    episode = soul.begin_episode("turn-r", task_text="rename the config file")
    episode.seal(soul.OUTCOME_COMPLETE)
    reply = json.dumps({"lessons": [{"title": "t", "tags": ["files"],
                                     "keywords": [], "text": "when x: y"}]})
    chronicle.reflect(episode.episode_id, lambda prompt: reply)
    assert spirit.state()["registers"]["clarity"] == 1


def test_an_applied_dream_rests_the_heart(monkeypatch):
    monkeypatch.setenv("HERMES_SPIRIT", "1")
    chronicle.chronicle_dir().mkdir(parents=True, exist_ok=True)
    for n in (1, 2):
        chronicle.add({"id": f"l{n}", "title": f"t{n}", "tags": ["x"],
                       "keywords": [], "text": f"when x{n}: y"})
    chronicle.apply_consolidation({
        "proposed_at": "ts", "shed": [],
        "merges": [{"absorb": ["l1", "l2"], "title": "kin", "text": "when x: y",
                    "tags": ["x"], "keywords": []}],
    })
    assert spirit.state()["registers"]["vigor"] == 2
