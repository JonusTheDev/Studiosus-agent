"""Tests for the Dyno live-throughput sampler seam in turn_finalizer."""

from types import SimpleNamespace
from unittest.mock import patch

from agent import turn_finalizer


def _agent(**kw):
    base = dict(model="qwen3.6:latest", base_url="http://localhost:11434/v1",
                _turn_gen_tokens=200, _turn_gen_seconds=4.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_records_sample_for_local_model():
    agent = _agent()
    with patch("agent.dyno.load_profile", return_value={"chosen_num_ctx": 65536}), \
         patch("agent.dyno.operating_num_ctx", return_value=65536), \
         patch("agent.dyno_history.record_sample", return_value=True) as rec:
        assert turn_finalizer.record_live_throughput_sample(agent) is True
    kwargs = rec.call_args.kwargs
    assert rec.call_args.args[0] == "qwen3.6:latest"
    assert kwargs["tok_s"] == 50.0          # 200 tokens / 4.0 s
    assert kwargs["num_ctx"] == 65536
    assert kwargs["source"] == "wallclock"


def test_no_sample_for_remote_model():
    # No profile and a non-local base_url -> not local -> no sample.
    agent = _agent(base_url="https://api.openai.com/v1")
    with patch("agent.dyno.load_profile", return_value=None), \
         patch("agent.dyno_history.record_sample") as rec:
        assert turn_finalizer.record_live_throughput_sample(agent) is False
        rec.assert_not_called()


def test_no_sample_when_no_generation_this_turn():
    agent = _agent(_turn_gen_tokens=0, _turn_gen_seconds=0.0)
    with patch("agent.dyno_history.record_sample") as rec:
        assert turn_finalizer.record_live_throughput_sample(agent) is False
        rec.assert_not_called()


def test_local_by_profile_even_if_base_url_absent():
    agent = _agent(base_url="")
    with patch("agent.dyno.load_profile", return_value={"power": {}}), \
         patch("agent.dyno.operating_num_ctx", return_value=None), \
         patch("agent.dyno_history.record_sample", return_value=True) as rec:
        assert turn_finalizer.record_live_throughput_sample(agent) is True
        rec.assert_called_once()


def test_sampler_never_raises_through_finalize_guard():
    # The finalize_turn call site wraps this in try/except; the helper itself
    # should also be robust. A record_sample explosion propagates here (callers
    # guard), but a missing attr must not — getattr defaults cover it.
    agent = SimpleNamespace(model="m:1")  # no gen counters at all
    assert turn_finalizer.record_live_throughput_sample(agent) is False
