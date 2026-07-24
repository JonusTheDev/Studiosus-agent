"""Tests for agent/dyno_report.py — the model-details rail payload."""

from unittest.mock import patch

from agent import dyno_report


_PROFILE = {
    "model": "benched:latest",
    "chosen_num_ctx": 65536,
    "chosen_because": "room to think",
    "power": {
        "recommended_num_ctx": 32768,
        "recommended_tok_s": 49.3,
        "recommended_in_band": True,
        "fully_fits": True,
        "curve": [
            {"num_ctx": 32768, "tok_s": 49.3, "prompt_tok_s": 900.0,
             "resident_gb": 12.0, "gpu_frac": 0.99, "load_s": 3.0},
            {"num_ctx": 65536, "tok_s": 50.2, "prompt_tok_s": 880.0,
             "resident_gb": 13.0, "gpu_frac": 0.95, "load_s": 3.2},
        ],
    },
}


def test_kpi_reads_64k_curve_point():
    kpi = dyno_report._kpi(_PROFILE)
    assert kpi["tok_s_at_64k"] == 50.2
    assert kpi["tok_s_at_32k"] == 49.3
    assert kpi["recommended_num_ctx"] == 32768
    assert kpi["fully_fits"] is True


def test_kpi_64k_none_when_point_absent():
    prof = {"power": {"curve": [{"num_ctx": 32768, "tok_s": 49.3}]}}
    assert dyno_report._kpi(prof)["tok_s_at_64k"] is None


def test_kpi_empty_profile_all_none():
    kpi = dyno_report._kpi(None)
    assert kpi["tok_s_at_64k"] is None
    assert kpi["recommended_num_ctx"] is None


def test_specialties_maps_ollama_capabilities():
    show = {"capabilities": ["completion", "tools", "vision", "thinking"]}
    with patch("agent.dyno._http", return_value=show):
        specs = dyno_report._specialties("m:1", "http://localhost:11434")
    # completion is the base capability -> no badge; thinking -> reasoning.
    assert specs == ["tools", "vision", "reasoning"]


def test_specialties_empty_on_unreachable():
    with patch("agent.dyno._http", side_effect=OSError("down")):
        assert dyno_report._specialties("m:1", "http://localhost:11434") == []


def test_unreachable_host_returns_empty_not_error():
    with patch("agent.dyno.installed_models", return_value=None):
        report = dyno_report.build_model_report(host="http://localhost:11434")
    assert report["reachable"] is False
    assert report["models"] == []
    assert report["flame"] is None


def test_build_joins_all_sources():
    installed = ["benched:latest", "bare:latest"]

    def _fake_profile(model):
        return _PROFILE if model == "benched:latest" else None

    def _fake_inspect(model, host=None):
        return {"loaded": model == "benched:latest", "pinned": True,
                "gpu_frac": 0.95, "loaded_num_ctx": 65536, "ready": True}

    with patch("agent.dyno.installed_models", return_value=installed), \
         patch("agent.dyno.load_profile", side_effect=_fake_profile), \
         patch("agent.flame.inspect", side_effect=_fake_inspect), \
         patch("agent.dyno_report._specialties", return_value=["tools", "vision"]), \
         patch("agent.dyno_history.summary",
               return_value={"samples": 5, "recent_tok_s_avg": 48.0,
                             "min": 40.0, "max": 55.0, "trend": [48.0]}):
        report = dyno_report.build_model_report(
            host="http://localhost:11434", current_model="benched:latest")

    assert report["reachable"] is True
    assert report["current_model"] == "benched:latest"
    assert report["flame"] is not None

    by_name = {m["model"]: m for m in report["models"]}
    # Benched model first (sort: benched before non-benched).
    assert report["models"][0]["model"] == "benched:latest"

    benched = by_name["benched:latest"]
    assert benched["benched"] is True
    assert benched["operating_num_ctx"] == 65536   # chosen_num_ctx wins
    assert benched["kpi"]["tok_s_at_64k"] == 50.2
    assert benched["profile"] is not None          # full curve for expansion
    assert benched["specialties"] == ["tools", "vision"]
    assert benched["live"]["samples"] == 5
    assert benched["status"]["pinned"] is True

    bare = by_name["bare:latest"]
    assert bare["benched"] is False
    assert bare["operating_num_ctx"] is None
    assert bare["kpi"]["tok_s_at_64k"] is None
    assert bare["profile"] is None


def test_build_never_raises_on_inspect_failure():
    with patch("agent.dyno.installed_models", return_value=["m:1"]), \
         patch("agent.dyno.load_profile", return_value=None), \
         patch("agent.flame.inspect", side_effect=RuntimeError("boom")), \
         patch("agent.dyno_report._specialties", return_value=[]), \
         patch("agent.dyno_history.summary", return_value=None):
        report = dyno_report.build_model_report(host="http://localhost:11434")
    assert report["models"][0]["status"]["loaded"] is False
