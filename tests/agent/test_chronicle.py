"""The Chronicle: distilling episodes into lessons, and serving them back.

Phase B of the Studiosus graft. The model call is injected, so the whole
reflect → shelve → retrieve → reinforce loop runs without a model.
"""

import json

import pytest

from agent import chronicle, soul


@pytest.fixture
def soul_on(monkeypatch):
    monkeypatch.setenv("HERMES_SOUL", "1")


def _sealed_episode(outcome=soul.OUTCOME_COMPLETE):
    episode = soul.begin_episode("turn-1", task_id="t", session_id="s",
                                 task_text="rename the config file")
    episode.record(soul.TOOL_CALL, tool="write_file", arguments='{"path": "a b.txt"}')
    episode.record(soul.TOOL_RESULT, tool="write_file", result="ok")
    episode.seal(outcome)
    return episode.episode_id


def _reply(*lessons):
    return json.dumps({"lessons": list(lessons)})


LESSON = {
    "title": "Quote paths with spaces",
    "tags": ["files", "shell"],
    "keywords": ["path", "quote", "rename"],
    "text": ("when writing to a path containing spaces: quote it, or the tool "
             "reports success while writing nothing"),
}


# --- reflection ----------------------------------------------------------


def test_a_sealed_episode_yields_shelved_lessons(soul_on):
    episode_id = _sealed_episode()
    shelved = chronicle.reflect(episode_id, lambda prompt: _reply(LESSON))
    assert len(shelved) == 1
    assert chronicle.entries()[0]["text"] == LESSON["text"]
    assert chronicle.entries()[0]["source_episode"] == episode_id
    assert chronicle.entries()[0]["source_outcome"] == soul.OUTCOME_COMPLETE


def test_an_honest_empty_reflection_shelves_nothing(soul_on):
    chronicle.reflect(_sealed_episode(), lambda prompt: _reply())
    assert chronicle.entries() == []


def test_no_more_than_three_lessons_survive_one_episode(soul_on):
    many = [dict(LESSON, title=f"lesson {i}", text=f"when x{i}: y") for i in range(6)]
    shelved = chronicle.reflect(_sealed_episode(), lambda prompt: _reply(*many))
    assert len(shelved) == chronicle.MAX_LESSONS_PER_EPISODE


def test_the_past_keeps_its_one_meaning(soul_on):
    # Reflecting twice on one episode must not double its lessons.
    episode_id = _sealed_episode()
    chronicle.reflect(episode_id, lambda prompt: _reply(LESSON))
    again = chronicle.reflect(episode_id, lambda prompt: _reply(LESSON))
    assert again == []
    assert len(chronicle.entries()) == 1


def test_an_unsealed_episode_is_not_yet_a_memory(soul_on):
    episode = soul.begin_episode("turn-x")
    episode.record(soul.THOUGHT, text="mid-flight")
    assert chronicle.reflect(episode.episode_id, lambda prompt: _reply(LESSON)) == []


def test_a_missing_episode_is_survivable(soul_on):
    assert chronicle.reflect("no-such-episode", lambda prompt: _reply(LESSON)) == []


@pytest.mark.parametrize("reply", [
    "", "not json at all", "{", '{"lessons": "not a list"}', '{"other": []}',
])
def test_malformed_reflection_shelves_nothing_and_raises_nothing(soul_on, reply):
    assert chronicle.reflect(_sealed_episode(), lambda prompt: reply) == []
    assert chronicle.entries() == []


def test_a_model_that_thinks_out_loud_is_still_understood(soul_on):
    # Small local models fence their JSON or narrate before it.
    reply = f"<think>Let me consider.</think>\nSure:\n```json\n{_reply(LESSON)}\n```"
    assert len(chronicle.reflect(_sealed_episode(), lambda prompt: reply)) == 1


def test_a_reflection_that_raises_is_a_quiet_evening(soul_on, caplog):
    def _boom(prompt):
        raise RuntimeError("aux model unreachable")

    assert chronicle.reflect(_sealed_episode(), _boom) == []
    # Quiet, but never silent: a reflection that produces nothing must be
    # distinguishable from a misconfiguration that produces nothing.
    assert "aux model unreachable" in caplog.text


def test_reflection_falls_back_to_the_model_that_did_the_work(monkeypatch):
    # Measured need: auxiliary auto-detect resolved to a model that was not
    # installed, so every reflection failed while the model that had just
    # finished the turn sat loaded and idle.
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if kwargs.get("task") == "reflection":
            raise RuntimeError("model 'gemma3:4b' not found")
        return "ok"

    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "call_llm", fake_call_llm)
    monkeypatch.setattr(aux, "extract_content_or_reasoning", lambda r: r)

    got = chronicle._auxiliary_call("prompt", runtime={
        "model": "qwen3.6:latest", "provider": "custom",
        "base_url": "http://127.0.0.1:11434/v1", "api_key": "ollama"})

    assert got == "ok"
    assert calls[0]["task"] == "reflection"          # configured path tried first
    assert calls[1]["model"] == "qwen3.6:latest"     # then the session's own model


def test_without_a_runtime_there_is_nothing_to_fall_back_to(monkeypatch):
    import agent.auxiliary_client as aux

    def _boom(**kwargs):
        raise RuntimeError("no aux configured")

    monkeypatch.setattr(aux, "call_llm", _boom)
    monkeypatch.setattr(aux, "extract_content_or_reasoning", lambda r: r)
    with pytest.raises(RuntimeError):
        chronicle._auxiliary_call("prompt", runtime=None)


def test_a_lesson_without_text_is_not_a_lesson(soul_on):
    shelved = chronicle.reflect(
        _sealed_episode(),
        lambda prompt: _reply({"title": "empty", "text": "   "}, LESSON))
    assert len(shelved) == 1
    assert shelved[0]["text"] == LESSON["text"]


def test_failed_episodes_teach_too(soul_on):
    # A fumble is exactly the kind of thing worth carrying forward.
    episode_id = _sealed_episode(outcome=soul.OUTCOME_FAILED)
    shelved = chronicle.reflect(episode_id, lambda prompt: _reply(LESSON))
    assert shelved[0]["source_outcome"] == soul.OUTCOME_FAILED


# --- retrieval -----------------------------------------------------------


def _shelve(**overrides):
    lesson = {"id": overrides.pop("id", "L1"), "title": "t", "text": "when x: y",
              "tags": [], "keywords": [], **overrides}
    chronicle.add(lesson)
    return lesson


def test_a_matching_lesson_is_served():
    _shelve(id="L1", tags=["files"], keywords=["rename"])
    assert [l["id"] for l in chronicle.retrieve("help me rename a file")] == ["L1"]


def test_an_irrelevant_lesson_is_not_served():
    _shelve(id="L1", tags=["kubernetes"], keywords=["helm"])
    # Silence beats spending the model's attention on a different situation.
    assert chronicle.retrieve("write me a poem about the sea") == []


def test_tags_weigh_more_than_keywords():
    _shelve(id="keyword-only", keywords=["deploy"])
    _shelve(id="tagged", tags=["deploy"])
    assert chronicle.retrieve("deploy the service")[0]["id"] == "tagged"


def test_only_k_lessons_are_laid_beside_the_work():
    for i in range(10):
        _shelve(id=f"L{i}", tags=["files"])
    assert len(chronicle.retrieve("files", k=2)) == 2
    assert len(chronicle.retrieve("files")) == chronicle.SERVE_K


def test_reinforcement_breaks_ties_but_never_outranks_a_better_match():
    _shelve(id="popular", tags=["files"])
    _shelve(id="better", tags=["files"], keywords=["rename", "config"])
    chronicle.reinforce(["popular"] * 25)
    # 'better' matches on a tag AND two keywords; no amount of past use may
    # promote the weaker match above it.
    assert chronicle.retrieve("rename the config files")[0]["id"] == "better"


def test_common_words_alone_do_not_match():
    _shelve(id="L1", tags=["the"], keywords=["with"])
    assert chronicle.retrieve("what should I do with the thing") == []


def test_an_empty_shelf_serves_nothing():
    assert chronicle.retrieve("anything at all") == []
    assert chronicle.render([]) == ""


def test_rendered_lessons_carry_their_text():
    lesson = _shelve(id="L1", tags=["files"], text="when renaming: quote the path")
    rendered = chronicle.render([lesson])
    assert "when renaming: quote the path" in rendered


# --- reinforcement -------------------------------------------------------


def test_serving_a_lesson_reinforces_it():
    chronicle.reinforce(["L1", "L2"])
    chronicle.reinforce(["L1"])
    assert chronicle.reinforcement() == {"L1": 2, "L2": 1}


def test_reinforcing_nothing_is_harmless():
    chronicle.reinforce([])
    chronicle.reinforce([None, ""])
    assert chronicle.reinforcement() == {}


def test_reinforcement_survives_a_corrupt_counts_file():
    chronicle.chronicle_dir().mkdir(parents=True, exist_ok=True)
    (chronicle.chronicle_dir() / "reinforcement.json").write_text("{ truncated")
    assert chronicle.reinforcement() == {}
    chronicle.reinforce(["L1"])
    assert chronicle.reinforcement() == {"L1": 1}


def test_a_corrupt_lesson_line_is_skipped_not_fatal():
    _shelve(id="L1", tags=["files"])
    with open(chronicle._lessons_path(), "a", encoding="utf-8") as f:
        f.write("{ this is not json\n")
    assert [l["id"] for l in chronicle.entries()] == ["L1"]


# --- the flag ------------------------------------------------------------


def test_it_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("HERMES_CHRONICLE", raising=False)
    assert chronicle.chronicle_enabled() is False
    monkeypatch.setenv("HERMES_CHRONICLE", "1")
    assert chronicle.chronicle_enabled() is True


# --- the pairing Phase D will need ---------------------------------------


def test_served_lessons_are_recorded_beside_the_outcome(soul_on):
    soul.record_turn(turn_id="turn-9", messages=[], final_response="done",
                     completed=True, served_lessons=["L1", "L2"])
    events = soul.read_episode(soul.list_episodes()[0])
    assemble = next(e for e in events if e["type"] == soul.ASSEMBLE)
    assert assemble["lessons"] == ["L1", "L2"]
    assert events[-1]["outcome"] == soul.OUTCOME_COMPLETE


def test_an_episode_that_was_served_nothing_records_an_empty_list(soul_on):
    soul.record_turn(turn_id="turn-10", messages=[], final_response="done",
                     completed=True)
    events = soul.read_episode(soul.list_episodes()[0])
    assemble = next(e for e in events if e["type"] == soul.ASSEMBLE)
    assert assemble["lessons"] == []
