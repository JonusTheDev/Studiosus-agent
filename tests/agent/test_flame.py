"""Tending the flame: what it reports, what it fixes, and what it refuses to.

Every test here runs against a fake Ollama — no server, no GPU.
"""

import pytest

from agent import dyno, flame


@pytest.fixture
def box(monkeypatch):
    """A fake Ollama whose resident set and reachability the test controls."""
    state = {
        "version": "0.30.8",
        "installed": ["qwen3.6:latest", "phi4:latest"],
        "resident": [],
        "calls": [],
    }

    monkeypatch.setattr(dyno, "server_version", lambda host=None: state["version"])
    monkeypatch.setattr(dyno, "installed_models", lambda host=None: state["installed"])
    monkeypatch.setattr(flame, "_resident", lambda host: list(state["resident"]))

    def fake_clear(host=None, keep=()):
        keep = set(keep)
        released = [m["model"] for m in state["resident"] if m["model"] not in keep]
        state["resident"] = [m for m in state["resident"] if m["model"] in keep]
        state["calls"].append(("clear", released))
        return released

    def fake_generate(model, prompt, num_ctx=None, **kwargs):
        state["calls"].append(("load", model, num_ctx, kwargs.get("keep_alive")))
        state["resident"] = [m for m in state["resident"] if m["model"] != model]
        state["resident"].append({
            "model": model, "context_length": num_ctx, "gpu_frac": 0.62,
            "pinned": kwargs.get("keep_alive") == flame.PIN_FOREVER,
        })
        return {}

    monkeypatch.setattr(dyno, "clear_field", fake_clear)
    monkeypatch.setattr(dyno, "generate_raw", fake_generate)
    monkeypatch.setattr(flame, "intended_num_ctx", lambda model: 65536)
    return state


def _resident(model, ctx=65536, pinned=True):
    return {"model": model, "context_length": ctx, "gpu_frac": 0.62, "pinned": pinned}


# --- looking ------------------------------------------------------------


def test_a_correctly_loaded_flame_is_ready(box):
    box["resident"] = [_resident("qwen3.6:latest")]
    report = flame.inspect("qwen3.6:latest")
    assert report["ready"] is True
    assert report["blockers"] == []
    assert "ready" in flame.describe(report)


def test_an_unloaded_model_is_not_ready(box):
    report = flame.inspect("qwen3.6:latest")
    assert report["ready"] is False
    assert any("not resident" in b for b in report["blockers"])


def test_the_wrong_context_is_not_ready(box):
    box["resident"] = [_resident("qwen3.6:latest", ctx=8192)]
    report = flame.inspect("qwen3.6:latest")
    assert report["ready"] is False
    assert any("wanted 65536" in b for b in report["blockers"])


def test_a_neighbour_holding_the_card_is_a_conflict(box):
    box["resident"] = [_resident("qwen3.6:latest"), _resident("phi4:latest")]
    report = flame.inspect("qwen3.6:latest")
    assert report["conflicts"] == ["phi4:latest"]
    assert report["ready"] is False


def test_inspect_changes_nothing(box):
    box["resident"] = [_resident("phi4:latest")]
    flame.inspect("qwen3.6:latest")
    assert box["calls"] == []


def test_an_unreachable_server_is_reported_not_guessed(box):
    box["version"] = None
    report = flame.inspect("qwen3.6:latest")
    assert report["reachable"] is False
    assert report["ready"] is False
    assert any("no Ollama answering" in b for b in report["blockers"])


def test_could_not_ask_is_not_the_same_as_not_installed(box, monkeypatch):
    # A network blip must not be reported as a missing model.
    monkeypatch.setattr(dyno, "installed_models", lambda host=None: None)
    report = flame.inspect("qwen3.6:latest")
    assert report["installed"] is None
    assert not any("not installed" in b for b in report["blockers"])


def test_with_no_profile_and_no_argument_there_is_no_wanted_context(box, monkeypatch):
    monkeypatch.setattr(flame, "intended_num_ctx", lambda model: None)
    box["resident"] = [_resident("qwen3.6:latest", ctx=4096)]
    report = flame.inspect("qwen3.6:latest")
    # No opinion means no complaint: we do not invent a number to judge against.
    assert report["want_num_ctx"] is None
    assert report["ready"] is True


# --- tending ------------------------------------------------------------


def test_a_cold_flame_is_loaded_at_the_chosen_context_and_pinned(box):
    report = flame.ensure_ready("qwen3.6:latest")
    assert ("load", "qwen3.6:latest", 65536, flame.PIN_FOREVER) in box["calls"]
    assert report["ready"] is True
    assert any("loaded" in a for a in report["actions"])


def test_conflicts_are_released_only_when_asked(box):
    box["resident"] = [_resident("phi4:latest")]
    flame.ensure_ready("qwen3.6:latest", clear_conflicts=False)
    assert not any(c[0] == "clear" for c in box["calls"])

    box["calls"].clear()
    box["resident"] = [_resident("phi4:latest")]
    report = flame.ensure_ready("qwen3.6:latest", clear_conflicts=True)
    assert ("clear", ["phi4:latest"]) in box["calls"]
    assert report["ready"] is True


def test_clearing_the_field_spares_the_flame_itself(box):
    box["resident"] = [_resident("qwen3.6:latest"), _resident("phi4:latest")]
    flame.ensure_ready("qwen3.6:latest", clear_conflicts=True)
    released = next(c[1] for c in box["calls"] if c[0] == "clear")
    assert released == ["phi4:latest"]


def test_a_model_resident_at_the_wrong_context_is_reloaded(box):
    box["resident"] = [_resident("qwen3.6:latest", ctx=8192)]
    report = flame.ensure_ready("qwen3.6:latest")
    assert ("load", "qwen3.6:latest", 65536, flame.PIN_FOREVER) in box["calls"]
    assert report["ready"] is True


def test_an_already_ready_flame_is_left_alone(box):
    box["resident"] = [_resident("qwen3.6:latest")]
    report = flame.ensure_ready("qwen3.6:latest")
    assert box["calls"] == []
    assert report["actions"] == []
    assert report["ready"] is True


# --- what it refuses to do ----------------------------------------------


def test_an_uninstalled_model_is_never_pulled_behind_your_back(box):
    box["installed"] = ["phi4:latest"]
    report = flame.ensure_ready("qwen3.6:latest", clear_conflicts=True)
    assert box["calls"] == []  # no load, and no field cleared for a model we lack
    assert report["ready"] is False
    assert any("not installed" in b for b in report["blockers"])


def test_an_unreachable_server_is_not_something_it_tries_to_start(box):
    box["version"] = None
    report = flame.ensure_ready("qwen3.6:latest", clear_conflicts=True)
    assert box["calls"] == []
    assert report["ready"] is False


def test_tending_never_raises_even_when_the_server_misbehaves(box, monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection reset mid-load")

    monkeypatch.setattr(dyno, "generate_raw", _boom)
    report = flame.ensure_ready("qwen3.6:latest")
    # Reported as a cold flame, not raised into the caller's startup.
    assert report["ready"] is False
    assert any("failed while tending" in a for a in report["actions"])


def test_it_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("HERMES_FLAME", raising=False)
    assert flame.flame_enabled() is False
    monkeypatch.setenv("HERMES_FLAME", "1")
    assert flame.flame_enabled() is True


def test_the_wanted_context_comes_from_the_profiles_choice(monkeypatch):
    monkeypatch.setattr(dyno, "load_profile", lambda model: {
        "model": model, "chosen_num_ctx": 65536,
        "power": {"recommended_num_ctx": 32768}})
    assert flame.intended_num_ctx("qwen3.6:latest") == 65536


def test_an_unmeasured_model_yields_no_wanted_context(monkeypatch):
    monkeypatch.setattr(dyno, "load_profile", lambda model: None)
    assert flame.intended_num_ctx("whatever:latest") is None


@pytest.mark.parametrize("base_url, expected", [
    # Hermes talks OpenAI-shaped; /api/ps and /api/generate hang off the root.
    ("http://localhost:11434/v1", "http://localhost:11434"),
    ("http://localhost:11434/v1/", "http://localhost:11434"),
    ("http://localhost:11434/api", "http://localhost:11434"),
    ("http://localhost:11434", "http://localhost:11434"),
    ("", None),
    (None, None),
])
def test_the_ollama_root_is_found_beneath_an_openai_base_url(base_url, expected):
    assert flame.host_from_base_url(base_url) == expected
