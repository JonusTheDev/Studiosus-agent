"""Tests for the dyno bench (agent/governor_dyno.py).

The sweep/aggregation is pure — the live roller is injected — so these run with
a scripted ``measure_fn`` and no GPU, no Ollama, no network.
"""

import agent.governor_dyno as dyno


def _flat_measure(**metrics):
    """A measure_fn that always returns the same point."""
    def measure(model, num_ctx, **kw):
        return dict(metrics)
    return measure


def test_builds_curve_and_recommendation():
    def measure(model, nc, **kw):
        # In band up to 8192, spilling past it.
        on_gpu = 0.98 if nc <= 8192 else 0.5
        return {"resident_gb": 12.0, "gpu_frac": on_gpu, "tok_s": 50.0,
                "prompt_tok_s": 400.0, "load_s": 2.0}
    power = dyno.run_dyno("m", measure, base_url="http://h/v1",
                          ctx_points=(4096, 8192, 16384), write=False)
    assert len(power["curve"]) == 3
    assert power["recommended_num_ctx"] == 8192  # deepest in-band
    assert power["recommended_in_band"] is True
    assert power["host"] == "http://h/v1"
    assert power["dyno_version"] == dyno.DYNO_VERSION


def test_mean_over_repeats():
    seq = iter([40.0, 60.0])  # two rolls → mean 50.0

    def measure(model, nc, **kw):
        return {"resident_gb": 10.0, "gpu_frac": 0.95, "tok_s": next(seq),
                "prompt_tok_s": 0.0, "load_s": 0.0}
    power = dyno.run_dyno("m", measure, base_url="http://h/v1",
                          ctx_points=(4096,), repeats=2, write=False)
    pt = power["curve"][0]
    assert pt["tok_s"] == 50.0
    assert pt["rolls"] == 2


def test_drops_error_rolls_from_mean():
    seq = iter([None, {"resident_gb": 10.0, "gpu_frac": 0.95, "tok_s": 55.0,
                       "prompt_tok_s": 0.0, "load_s": 0.0}])

    def measure(model, nc, **kw):
        return next(seq)
    power = dyno.run_dyno("m", measure, base_url="http://h/v1",
                          ctx_points=(4096,), repeats=2, write=False)
    pt = power["curve"][0]
    assert pt["tok_s"] == 55.0  # the None roll dropped
    assert pt["rolls"] == 1


def test_point_skipped_when_all_rolls_fail():
    def measure(model, nc, **kw):
        return None  # model cannot run this context at all
    power = dyno.run_dyno("m", measure, base_url="http://h/v1",
                          ctx_points=(4096, 8192), write=False)
    assert power["curve"] == []
    assert power["recommended_num_ctx"] is None
    assert power["recommended_in_band"] is False


def test_num_gpu_points_cross_each_context():
    seen = []

    def measure(model, nc, **kw):
        seen.append((nc, kw.get("num_gpu")))
        return {"resident_gb": 10.0, "gpu_frac": 0.95, "tok_s": 50.0,
                "prompt_tok_s": 0.0, "load_s": 0.0}
    power = dyno.run_dyno("m", measure, base_url="http://h/v1",
                          ctx_points=(4096,), num_gpu_points=(20, 41), write=False)
    assert (4096, 20) in seen and (4096, 41) in seen
    assert all("num_gpu" in p for p in power["curve"])


def test_write_persists_via_save_curve(monkeypatch):
    saved = {}
    import agent.governor as governor
    monkeypatch.setattr(governor, "save_curve",
                        lambda model, base_url, power: saved.update(
                            {"model": model, "base_url": base_url, "power": power}))
    dyno.run_dyno("qwen3:8b", _flat_measure(resident_gb=10.0, gpu_frac=0.95, tok_s=50.0,
                                            prompt_tok_s=0.0, load_s=0.0),
                  base_url="http://h/v1", ctx_points=(4096,), write=True)
    assert saved["model"] == "qwen3:8b" and saved["base_url"] == "http://h/v1"
    assert saved["power"]["curve"]


def test_no_write_does_not_persist(monkeypatch):
    called = {"n": 0}
    import agent.governor as governor
    monkeypatch.setattr(governor, "save_curve", lambda *a, **k: called.update(n=called["n"] + 1))
    dyno.run_dyno("m", _flat_measure(resident_gb=10.0, gpu_frac=0.95, tok_s=50.0,
                                     prompt_tok_s=0.0, load_s=0.0),
                  base_url="http://h/v1", ctx_points=(4096,), write=False)
    assert called["n"] == 0


def test_below_band_recommendation_flagged_not_in_band():
    def measure(model, nc, **kw):
        # Nothing holds the 0.90 band; fastest is the shallowest.
        return {"resident_gb": 20.0, "gpu_frac": 0.5,
                "tok_s": 60.0 if nc == 4096 else 40.0,
                "prompt_tok_s": 0.0, "load_s": 0.0}
    power = dyno.run_dyno("m", measure, base_url="http://h/v1",
                          ctx_points=(4096, 8192), write=False)
    assert power["recommended_num_ctx"] == 4096  # steered by tok/s below band
    assert power["recommended_in_band"] is False
