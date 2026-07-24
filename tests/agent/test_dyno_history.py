"""Tests for agent/dyno_history.py — the live-throughput sampler."""

from agent import dyno_history


def test_record_and_summary_roundtrip():
    for t in (40.0, 50.0, 60.0):
        assert dyno_history.record_sample(
            "qwen3.6:latest", tok_s=t, num_ctx=65536,
            source="wallclock", completion_tokens=100) is True
    s = dyno_history.summary("qwen3.6:latest")
    assert s is not None
    assert s["samples"] == 3
    assert s["min"] == 40.0
    assert s["max"] == 60.0
    assert s["recent_tok_s_avg"] == 50.0
    assert s["trend"] == [40.0, 50.0, 60.0]
    assert s["last_source"] == "wallclock"
    assert s["last_seen"]


def test_summary_none_when_no_samples():
    assert dyno_history.summary("never:seen") is None


def test_record_rejects_nonpositive_tok_s():
    assert dyno_history.record_sample("m:1", tok_s=0) is False
    assert dyno_history.record_sample("m:1", tok_s=-5) is False
    assert dyno_history.summary("m:1") is None


def test_record_rejects_empty_model():
    assert dyno_history.record_sample("", tok_s=42.0) is False


def test_trim_keeps_only_most_recent(monkeypatch):
    monkeypatch.setattr(dyno_history, "MAX_SAMPLES", 5)
    for i in range(20):
        # vary tok_s so we can assert which survived
        dyno_history.record_sample("trim:me", tok_s=float(i + 1))
    samples = dyno_history.read_samples("trim:me")
    assert len(samples) == 5
    # The last five recorded (16..20) survived; the head was trimmed.
    assert [s["tok_s"] for s in samples] == [16.0, 17.0, 18.0, 19.0, 20.0]


def test_recent_average_spans_only_recent_window(monkeypatch):
    monkeypatch.setattr(dyno_history, "RECENT_WINDOW", 3)
    for t in (10.0, 10.0, 100.0, 100.0, 100.0):
        dyno_history.record_sample("recent:win", tok_s=t)
    s = dyno_history.summary("recent:win")
    # recent avg is the last 3 (all 100), not the full mean.
    assert s["recent_tok_s_avg"] == 100.0
    # min/max still span the whole retained window.
    assert s["min"] == 10.0
    assert s["max"] == 100.0


def test_corrupt_line_is_skipped():
    dyno_history.record_sample("corrupt:me", tok_s=42.0)
    path = dyno_history._path("corrupt:me")
    with open(path, "a", encoding="utf-8") as f:
        f.write("this is not json\n")
    dyno_history.record_sample("corrupt:me", tok_s=44.0)
    samples = dyno_history.read_samples("corrupt:me")
    assert [s["tok_s"] for s in samples] == [42.0, 44.0]


def test_record_never_raises(monkeypatch):
    # Force the write path to explode; record_sample must swallow it.
    def _boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(dyno_history.Path, "mkdir", _boom)
    assert dyno_history.record_sample("m:1", tok_s=42.0) is False
