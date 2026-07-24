"""The Soul — Phase A of the Studiosus graft.

A finished turn must seal exactly one episode, stamped with the right outcome,
and the ledger must never be able to break the turn it observes.  See
``.plans/studiosus-heart-strategy.md``.
"""

import json

import pytest

from agent import soul
from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 1
    max_total = 10
    remaining = 9


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from.

    Mirrors ``tests/agent/test_turn_finalizer_cleanup_guard.py``; kept local so
    a change to that regression test can't quietly reshape this one.
    """

    def __init__(self):
        self.max_iterations = 10
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _messages():
    """One prior turn, then this turn: a thought, a tool call, its result."""
    return [
        {"role": "user", "content": "an earlier question"},
        {"role": "assistant", "content": "an earlier answer"},
        {"role": "user", "content": "read the config"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "I should read the file first.",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": '{"path": "cfg"}'}}
            ],
        },
        {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": "port = 8080"},
    ]


def _run(agent, *, final_response="the port is 8080", interrupted=False, failed=False,
         exit_reason="text_response(finish_reason=stop)", api_call_count=2):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=_messages(),
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="sess-1:task-1:abcd1234",
        user_message="read the config",
        original_user_message="read the config",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
    )


@pytest.fixture
def soul_on(monkeypatch):
    monkeypatch.setenv("HERMES_SOUL", "1")


# --- the covenant --------------------------------------------------------


def test_completed_turn_seals_exactly_one_episode_with_complete_outcome(soul_on):
    result = _run(_StubAgent())
    assert result["completed"] is True

    episodes = soul.list_episodes()
    assert len(episodes) == 1

    events = soul.read_episode(episodes[0])
    assert soul.is_sealed(episodes[0])
    assert events[-1]["type"] == soul.EPISODE_END
    assert events[-1]["outcome"] == soul.OUTCOME_COMPLETE

    # seq is monotonic from 1, so ordering never rests on timestamp resolution
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


def test_episode_records_the_turn_and_not_the_one_before_it(soul_on):
    _run(_StubAgent())
    events = soul.read_episode(soul.list_episodes()[0])
    by_type = {}
    for event in events:
        by_type.setdefault(event["type"], []).append(event)

    assert by_type[soul.EPISODE_BEGIN][0]["task_id"] == "task-1"
    assert by_type[soul.EPISODE_BEGIN][0]["session_id"] == "sess-1"
    assert by_type[soul.THOUGHT][0]["text"] == "I should read the file first."
    assert by_type[soul.TOOL_CALL][0]["tool"] == "read_file"
    assert by_type[soul.TOOL_RESULT][0]["result"] == "port = 8080"
    assert by_type[soul.DELIVERY][0]["text"] == "the port is 8080"

    # the prior turn is a different memory and must not be woven into this one
    assert not any("earlier" in json.dumps(e) for e in events)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"failed": True}, soul.OUTCOME_FAILED),
        ({"interrupted": True}, soul.OUTCOME_ABANDONED),
        ({"final_response": None, "exit_reason": "unknown"}, soul.OUTCOME_INCOMPLETE),
    ],
)
def test_a_turn_seals_with_the_outcome_it_actually_had(soul_on, kwargs, expected):
    _run(_StubAgent(), **kwargs)
    events = soul.read_episode(soul.list_episodes()[0])
    assert events[-1]["outcome"] == expected


def test_an_interrupted_failure_is_abandoned_not_failed():
    # Interruption is checked first: the user calling a turn off is not the
    # same event as the turn breaking, and the ledger must not conflate them.
    assert soul.turn_outcome(
        completed=False, failed=True, interrupted=True
    ) == soul.OUTCOME_ABANDONED


def test_sealed_episode_refuses_further_writes(soul_on):
    episode = soul.begin_episode("turn-x", task_id="t", session_id="s")
    episode.record(soul.THOUGHT, text="a thought")
    episode.seal(soul.OUTCOME_COMPLETE)

    with pytest.raises(soul.SealedEpisodeError):
        episode.record(soul.THOUGHT, text="a revision of the past")

    events = soul.read_episode(episode.episode_id)
    assert [e["type"] for e in events] == [
        soul.EPISODE_BEGIN, soul.THOUGHT, soul.EPISODE_END
    ]


# --- it must not be able to hurt anything --------------------------------


def test_off_by_default_records_nothing():
    result = _run(_StubAgent())
    assert result["completed"] is True
    assert soul.list_episodes() == []


def test_a_broken_ledger_never_breaks_the_turn(soul_on, monkeypatch):
    def _explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(soul, "begin_episode", _explode)
    result = _run(_StubAgent())

    # The turn is untouched: the response survives and nothing is reported as
    # a cleanup failure, because the Soul is an observer, not a step.
    assert result["final_response"] == "the port is 8080"
    assert result["completed"] is True
    assert "cleanup_errors" not in result
    assert soul.list_episodes() == []


def test_secrets_are_redacted_before_they_reach_the_disk(soul_on):
    episode = soul.begin_episode("turn-y")
    episode.record(soul.TOOL_RESULT, tool="shell", result="export sk-ant-api03-" + "A" * 95)
    episode.seal(soul.OUTCOME_COMPLETE)
    raw = episode.path.read_text(encoding="utf-8")
    assert "sk-ant-api03-" + "A" * 95 not in raw


# --- the spirit hears the seal (Phase E wiring) --------------------------


def test_a_sealed_turn_moves_the_heart_when_the_spirit_is_on(soul_on, monkeypatch):
    monkeypatch.setenv("HERMES_SPIRIT", "1")
    from agent import spirit

    _run(_StubAgent())

    st = spirit.state()
    assert st["registers"]["joy"] == 1    # good toil...
    assert st["registers"]["vigor"] == -1  # ...but toil


def test_without_a_soul_no_memory_moves_the_heart(monkeypatch):
    monkeypatch.delenv("HERMES_SOUL", raising=False)
    monkeypatch.setenv("HERMES_SPIRIT", "1")
    from agent import spirit

    _run(_StubAgent())

    # No episode sealed means no memory was made: the registers stay at rest.
    assert spirit.state()["registers"]["joy"] == 0
