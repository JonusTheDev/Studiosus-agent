"""Tests for the local-Ollama Governor (agent/governor.py).

The judgment layer is pure (no GPU, no network) — ported faithfully from
Studiosus, so these mirror its mock tests. The orchestrator
(``resolve_local_num_ctx``) is exercised with the senses monkeypatched, so the
whole suite runs fast with no Ollama server and no CUDA.
"""

import agent.governor as gov


# ── recommend_num_ctx ──────────────────────────────────────────────────────────

def test_recommend_in_band_picks_deepest():
    curve = [
        {"num_ctx": 4096, "gpu_frac": 1.00, "tok_s": 60},
        {"num_ctx": 8192, "gpu_frac": 0.95, "tok_s": 55},
        {"num_ctx": 16384, "gpu_frac": 0.80, "tok_s": 40},  # below the 0.90 band
    ]
    assert gov.recommend_num_ctx(curve)["num_ctx"] == 8192


def test_recommend_in_band_ties_break_on_tok_s():
    curve = [
        {"num_ctx": 8192, "gpu_frac": 0.95, "tok_s": 40},
        {"num_ctx": 8192, "gpu_frac": 0.95, "tok_s": 55},  # same depth, faster
    ]
    assert gov.recommend_num_ctx(curve)["tok_s"] == 55


def test_recommend_below_band_picks_deepest_within_speed_tolerance():
    # No point holds the band; steer by measured tok/s, depth when speed-free.
    curve = [
        {"num_ctx": 4096, "gpu_frac": 0.70, "tok_s": 60},
        {"num_ctx": 8192, "gpu_frac": 0.65, "tok_s": 58},   # within 95% of 60
        {"num_ctx": 16384, "gpu_frac": 0.60, "tok_s": 40},  # too slow
    ]
    assert gov.recommend_num_ctx(curve)["num_ctx"] == 8192


def test_recommend_respects_ceiling():
    curve = [
        {"num_ctx": 4096, "gpu_frac": 1.00, "tok_s": 60},
        {"num_ctx": 8192, "gpu_frac": 0.95, "tok_s": 55},
    ]
    assert gov.recommend_num_ctx(curve, ceiling=4096)["num_ctx"] == 4096


def test_recommend_empty_curve_is_none():
    assert gov.recommend_num_ctx([]) is None


# ── gpu_fraction ────────────────────────────────────────────────────────────────

def test_gpu_fraction():
    assert gov.gpu_fraction(0, 0) == 0.0
    assert gov.gpu_fraction(100, 100) == 1.0
    assert gov.gpu_fraction(100, 50) == 0.5
    assert gov.gpu_fraction(100, 200) == 1.0  # clamped to 1.0


# ── check_plan ──────────────────────────────────────────────────────────────────

def test_check_plan_unknown_without_curve():
    assert gov.check_plan({"model": "m", "power": {}}, 8192)["status"] == "unknown"


def test_check_plan_in_band_and_spills():
    profile = {"model": "m", "power": {"curve": [
        {"num_ctx": 4096, "gpu_frac": 1.0, "tok_s": 60},
        {"num_ctx": 16384, "gpu_frac": 0.5, "tok_s": 30},
    ], "recommended_num_ctx": 4096}}
    assert gov.check_plan(profile, 4096)["status"] == "in_band"
    assert gov.check_plan(profile, 16384)["status"] == "spills"


# ── govern_options ──────────────────────────────────────────────────────────────

def test_govern_options_caps_to_recommendation_out_loud():
    profile = {"name": "m", "power": {"recommended_num_ctx": 8192}}
    logs = []
    out = gov.govern_options(profile, {"num_ctx": 32768}, log_fn=lambda t, m: logs.append(m))
    assert out["num_ctx"] == 8192
    assert logs, "capping must be logged, never silent"


def test_govern_options_does_not_mutate_input():
    profile = {"name": "m", "power": {"recommended_num_ctx": 8192}}
    original = {"num_ctx": 32768}
    gov.govern_options(profile, original, log_fn=lambda *a: None)
    assert original["num_ctx"] == 32768


def test_govern_options_decree_overrides_counsel():
    profile = {"name": "m", "num_ctx_decree": 65536, "power": {"recommended_num_ctx": 8192}}
    out = gov.govern_options(profile, {"num_ctx": 131072}, log_fn=lambda *a: None)
    assert out["num_ctx"] == 65536


def test_govern_options_no_curve_runs_blind_uncapped():
    profile = {"name": "m", "power": {}}
    out = gov.govern_options(profile, {"num_ctx": 131072}, log_fn=lambda *a: None)
    assert out["num_ctx"] == 131072  # flagged BLIND, but never silently capped


# ── preflight ───────────────────────────────────────────────────────────────────

def test_preflight_refuses_absent_model():
    r = gov.preflight({"model": "qwen3", "name": "qwen3"}, ["llama3:8b"])
    assert r["ok"] is False and "not installed" in r["reason"]


def test_preflight_unreachable_server_is_not_a_refusal():
    # installed=None means "server unreachable" — must NOT refuse (carved lesson).
    r = gov.preflight({"model": "qwen3", "name": "qwen3"}, None)
    assert r["ok"] is True


def test_preflight_present_without_curve_warns():
    r = gov.preflight({"model": "qwen3", "name": "qwen3"}, ["qwen3:latest"])
    assert r["ok"] is True and any("no power curve" in w for w in r["warnings"])


# ── _model_installed (tag matching) ─────────────────────────────────────────────

def test_model_installed_matching():
    assert gov._model_installed("qwen3", ["qwen3:latest"])
    assert gov._model_installed("qwen3:8b", ["qwen3:8b"])
    assert gov._model_installed("qwen3", ["qwen3:8b"])  # config omits tag; base-name match
    assert not gov._model_installed("mistral", ["qwen3:8b", "llama3:latest"])


# ── resolve_local_num_ctx (orchestrator, senses monkeypatched) ──────────────────

def _patch_senses(monkeypatch, *, server="ollama", installed=("qwen3:8b",),
                  split=(None, None), curve=None):
    import agent.model_metadata as mm
    monkeypatch.setattr(mm, "detect_local_server_type", lambda *a, **k: server)
    monkeypatch.setattr(gov, "installed_models", lambda *a, **k: (list(installed) if installed is not None else None))
    monkeypatch.setattr(gov, "model_split", lambda *a, **k: split)
    monkeypatch.setattr(gov, "load_curve", lambda *a, **k: curve)


def test_resolve_non_ollama_is_noop(monkeypatch):
    _patch_senses(monkeypatch, server="vllm")
    r = gov.resolve_local_num_ctx(model="x", base_url="http://h/v1", requested_ctx=131072)
    assert r.safe_ctx == 131072 and r.refusal is None and not r.warnings


def test_resolve_refuses_absent(monkeypatch):
    _patch_senses(monkeypatch, installed=["llama3:8b"])
    r = gov.resolve_local_num_ctx(model="qwen3", base_url="http://h/v1", requested_ctx=131072)
    assert r.refusal is not None and r.safe_ctx is None


def test_resolve_unreachable_passes_through(monkeypatch):
    _patch_senses(monkeypatch, installed=None)
    r = gov.resolve_local_num_ctx(model="qwen3", base_url="http://h/v1", requested_ctx=131072)
    assert r.refusal is None and r.safe_ctx == 131072


def test_resolve_caps_with_curve(monkeypatch):
    curve = {"curve": [{"num_ctx": 65536, "gpu_frac": 0.95, "tok_s": 40}],
             "recommended_num_ctx": 65536}
    _patch_senses(monkeypatch, installed=["qwen3:8b"], curve=curve)
    r = gov.resolve_local_num_ctx(model="qwen3:8b", base_url="http://h/v1", requested_ctx=262144)
    assert r.safe_ctx == 65536 and any("power band" in w for w in r.warnings)


def test_resolve_applies_context_length_ceiling(monkeypatch):
    _patch_senses(monkeypatch, installed=["qwen3:8b"])  # no curve
    r = gov.resolve_local_num_ctx(model="qwen3:8b", base_url="http://h/v1",
                                  requested_ctx=262144, ceiling=100_000)
    assert r.safe_ctx == 100_000 and any("context_length" in w for w in r.warnings)


def test_resolve_measured_spill_guard(monkeypatch):
    # Model resident and spilling: 20 GB total, only 10 GB on the GPU (0.5 frac).
    split = (20 * 1024 ** 3, 10 * 1024 ** 3)
    _patch_senses(monkeypatch, installed=["qwen3:8b"], split=split)
    r = gov.resolve_local_num_ctx(model="qwen3:8b", base_url="http://h/v1", requested_ctx=65536)
    assert any("spilling" in w for w in r.warnings)


def test_resolve_tool_use_floor_reconciliation(monkeypatch):
    # Curve's safe context (32K) is below Hermes's 64K tool-use floor: keep the
    # floor, and warn about the spill rather than silently dropping tools.
    curve = {"curve": [{"num_ctx": 32768, "gpu_frac": 0.95, "tok_s": 40}],
             "recommended_num_ctx": 32768}
    _patch_senses(monkeypatch, installed=["qwen3:8b"], curve=curve)
    r = gov.resolve_local_num_ctx(model="qwen3:8b", base_url="http://h/v1",
                                  requested_ctx=131072, tool_use_floor=64_000)
    assert r.safe_ctx == 64_000 and any("tool-use floor" in w for w in r.warnings)


def test_resolve_none_requested_ctx_still_refuses_absent(monkeypatch):
    # Detection returns None for an unpulled model; refuse-if-absent must still fire.
    _patch_senses(monkeypatch, installed=["llama3:8b"])
    r = gov.resolve_local_num_ctx(model="qwen3", base_url="http://h/v1", requested_ctx=None)
    assert r.refusal is not None


def test_resolve_none_requested_ctx_passthrough_when_present(monkeypatch):
    # Present model, no detected ctx, no curve → no crash, no cap, no refusal.
    _patch_senses(monkeypatch, installed=["qwen3:8b"])
    r = gov.resolve_local_num_ctx(model="qwen3:8b", base_url="http://h/v1", requested_ctx=None)
    assert r.refusal is None and r.safe_ctx is None


def test_resolve_decree_holds_over_curve(monkeypatch):
    # A user decree of 65536 must win over the curve's 8192 counsel.
    curve = {"curve": [{"num_ctx": 8192, "gpu_frac": 0.95, "tok_s": 55}],
             "recommended_num_ctx": 8192}
    _patch_senses(monkeypatch, installed=["qwen3:8b"], curve=curve)
    r = gov.resolve_local_num_ctx(model="qwen3:8b", base_url="http://h/v1",
                                  requested_ctx=131072, decree=65536)
    assert r.safe_ctx == 65536
