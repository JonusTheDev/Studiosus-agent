"""Phase C — consolidation under covenant.

The dream is counsel; apply_consolidation is law.  Every test here checks the
law, with dreams both honest and adversarial.
"""

import json

import pytest

from agent import chronicle


def _shelve(lid, *, tags=(), text=None, title=None):
    lesson = {"id": lid, "title": title or lid, "text": text or f"when {lid}: x",
              "tags": list(tags), "keywords": []}
    assert chronicle.add(lesson)
    return lesson


def _merge(*absorb, text="when either: sharpened", title="merged"):
    return {"title": title, "tags": ["merged"], "keywords": [],
            "text": text, "absorb": list(absorb)}


# --- the covenant, against an adversarial dream --------------------------


def test_a_reinforced_lesson_is_never_shed_whatever_the_dream_asked():
    _shelve("load-bearing")
    _shelve("platitude")
    chronicle.reinforce(["load-bearing"])

    summary = chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": ["load-bearing", "platitude"], "merges": []})

    assert [lesson["id"] for lesson in chronicle.entries()] == ["load-bearing"]
    assert summary["refused"] == [{"id": "load-bearing", "why": "reinforced; never shed"}]
    assert summary["released"] == 1


def test_shed_lessons_are_archived_never_burned():
    lesson = _shelve("platitude", text="when anything: be careful")
    chronicle.apply_consolidation({"proposed_at": "t1", "shed": ["platitude"],
                                   "merges": []})

    assert chronicle.entries() == []
    archived = [json.loads(line) for line in
                chronicle._shed_path().read_text(encoding="utf-8").splitlines()]
    assert archived[0]["id"] == "platitude"
    assert archived[0]["text"] == lesson["text"]  # readable back, whole
    assert archived[0]["shed_reason"] == "consolidation:t1"


def test_a_merge_absorbs_its_kin_and_archives_them():
    _shelve("a", tags=["files"])
    _shelve("b", tags=["files"])
    summary = chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": [], "merges": [_merge("a", "b")]})

    ids = [lesson["id"] for lesson in chronicle.entries()]
    assert ids == ["consolidated-t1-1"]
    assert chronicle.entries()[0]["absorbed"] == ["a", "b"]
    assert summary == {"before": 2, "after": 1, "merged": 1, "released": 2,
                       "refused": []}
    # absorbed lessons land in the shed archive too — merged away, not burned
    archived = {json.loads(line)["id"] for line in
                chronicle._shed_path().read_text(encoding="utf-8").splitlines()}
    assert archived == {"a", "b"}


def test_a_merge_inherits_the_reinforcement_of_what_it_absorbed():
    _shelve("a")
    _shelve("b")
    chronicle.reinforce(["a", "a", "b"])  # a=2, b=1

    chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": [], "merges": [_merge("a", "b")]})

    # 3 inherited: the merge is immediately as load-bearing as its parents,
    # so a later dream cannot shed it either.
    assert chronicle.reinforcement()["consolidated-t1-1"] == 3
    summary = chronicle.apply_consolidation(
        {"proposed_at": "t2", "shed": ["consolidated-t1-1"], "merges": []})
    assert summary["refused"][0]["why"] == "reinforced; never shed"


def test_a_reinforced_lesson_may_still_be_merged():
    # Reinforced protects against shedding, not against sharpening.
    _shelve("a")
    _shelve("b")
    chronicle.reinforce(["a"])
    summary = chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": [], "merges": [_merge("a", "b")]})
    assert summary["merged"] == 1


@pytest.mark.parametrize("bad_merge, why_fragment", [
    (_merge("a"), ">=2 known absorbed ids"),                      # one id only
    (_merge("a", "ghost"), ">=2 known absorbed ids"),             # unknown kin
    (_merge("a", "b", text="   "), ">=2 known absorbed ids"),     # no text
])
def test_a_malformed_merge_is_refused_and_absorbs_nothing(bad_merge, why_fragment):
    _shelve("a")
    _shelve("b")
    summary = chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": [], "merges": [bad_merge]})
    assert sorted(lesson["id"] for lesson in chronicle.entries()) == ["a", "b"]
    assert why_fragment in summary["refused"][0]["why"]


def test_unknown_shed_ids_are_refused_by_name():
    _shelve("a")
    summary = chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": ["never-existed"], "merges": []})
    assert summary["refused"] == [{"id": "never-existed", "why": "unknown id"}]
    assert len(chronicle.entries()) == 1


def test_an_untouched_lesson_survives_consolidation_verbatim():
    kept = _shelve("kept", tags=["files"], text="when x: exactly this")
    _shelve("shed-me")
    chronicle.apply_consolidation({"proposed_at": "t1", "shed": ["shed-me"],
                                   "merges": []})
    assert chronicle.entries() == [kept]


def test_a_lesson_shed_by_one_clause_cannot_also_be_absorbed():
    _shelve("a")
    _shelve("b")
    _shelve("c")
    # The dream sheds 'a' AND absorbs it: the shed wins (processed first) and
    # the merge then lacks two known ids, so it is refused rather than
    # double-counting 'a'.
    summary = chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": ["a"], "merges": [_merge("a", "b", "c")]})
    # a shed, b+c still eligible: merge of (b, c) proceeds
    assert summary["merged"] == 1
    assert chronicle.entries()[0]["absorbed"] == ["b", "c"]


# --- propose: the dream writes, and changes nothing ----------------------


def test_propose_writes_a_proposal_and_touches_no_lesson():
    _shelve("a", tags=["files"])
    _shelve("b", tags=["files"])
    before = chronicle.entries()

    got = chronicle.propose_consolidation(
        lambda prompt: json.dumps({"merges": [_merge("a", "b")], "shed": []}))

    assert chronicle.entries() == before           # nothing applied
    assert got["proposal"]["merges"][0]["absorb"] == ["a", "b"]
    assert chronicle.list_proposals()[0]["applied_at"] is None


def test_kin_sort_lands_shared_tags_in_one_batch():
    _shelve("z-files", tags=["files"])
    _shelve("a-net", tags=["network"])
    _shelve("a-files", tags=["files"])
    batches = []
    chronicle.propose_consolidation(
        lambda prompt: batches.append(prompt) or "{}", batch_size=2)
    # tag-sorted: the two files lessons share batch 1 despite their titles
    assert "a-files" in batches[0] and "z-files" in batches[0]
    assert "a-net" in batches[1]


def test_a_failed_batch_loses_its_suggestions_not_the_run(caplog):
    for i in range(4):
        _shelve(f"l{i}", tags=["t"])
    replies = iter([RuntimeError("model died"),
                    json.dumps({"merges": [], "shed": ["l2"]})])

    def flaky(prompt):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    got = chronicle.propose_consolidation(flaky, batch_size=2, write=False)
    assert got["proposal"]["shed"] == ["l2"]      # batch 2 still contributed
    assert "batch 1 failed" in caplog.text


def test_the_prompt_shows_the_dream_each_lessons_reinforcement():
    _shelve("a")
    chronicle.reinforce(["a", "a"])
    seen = []
    chronicle.propose_consolidation(lambda p: seen.append(p) or "{}", write=False)
    assert "id=a REINFORCED=2" in seen[0]


# --- the desk: apply exactly once ----------------------------------------


def test_a_desk_proposal_applies_exactly_once():
    _shelve("platitude")
    got = chronicle.propose_consolidation(
        lambda prompt: json.dumps({"merges": [], "shed": ["platitude"]}))
    ts = got["proposal"]["proposed_at"]

    summary = chronicle.apply_proposal_file(ts)
    assert summary["released"] == 1
    assert chronicle.list_proposals()[0]["applied_at"] is not None

    with pytest.raises(ValueError, match="already applied"):
        chronicle.apply_proposal_file(ts)


def test_applying_a_proposal_that_never_existed_raises():
    with pytest.raises(KeyError):
        chronicle.apply_proposal_file("no-such-ts")


def test_an_empty_shelf_dreams_an_empty_proposal():
    got = chronicle.propose_consolidation(lambda prompt: "{}", write=False)
    assert got["proposal"] == {"proposed_at": got["proposal"]["proposed_at"],
                               "lesson_count": 0, "merges": [], "shed": []}


# --- consolidated lessons keep working -----------------------------------


def test_a_consolidated_lesson_is_retrievable_like_any_other():
    _shelve("a", tags=["files"])
    _shelve("b", tags=["files"])
    chronicle.apply_consolidation(
        {"proposed_at": "t1", "shed": [],
         "merges": [_merge("a", "b", text="when handling files: the sharp truth")]})
    got = chronicle.retrieve("help with these files")
    assert [lesson["id"] for lesson in got] == ["consolidated-t1-1"]
