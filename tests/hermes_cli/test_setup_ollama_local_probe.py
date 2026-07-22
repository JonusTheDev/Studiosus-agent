"""Local Ollama is a fully first-class provider in ``hermes setup``.

Unlike the generic fallback path, the local ``ollama`` provider gets its own
branch in ``_model_flow_api_key_provider`` that live-probes the OpenAI-compatible
``/v1/models`` endpoint for the user's pulled models, prints a "Found N" line on
success, and — most importantly — guides the user (start the server / pull a
model) instead of silently dropping to raw model-name input when the local
server is offline.
"""

from __future__ import annotations

import inspect

import hermes_cli.model_setup_flows as flows_mod


def _ollama_branch_snippet() -> str:
    """Return only the local-ollama branch, bounded by the next elif so it
    never bleeds into the adjacent ollama-cloud branch."""
    src = inspect.getsource(flows_mod)
    marker = 'provider_id == "ollama"'
    assert marker in src, "local ollama branch missing from provider setup"
    idx = src.index(marker)
    rest = src[idx + len(marker):]
    end = rest.find("elif provider_id ==")
    return rest if end == -1 else rest[:end]


def test_ollama_branch_live_probes_local_models_endpoint():
    """The local ollama setup branch must probe the /v1/models endpoint."""
    snippet = _ollama_branch_snippet()
    assert "fetch_api_models(" in snippet, snippet[:400]
    assert "effective_base" in snippet, snippet[:400]
    assert 'Found ' in snippet and 'from Ollama' in snippet, snippet[:400]


def test_ollama_branch_guides_user_when_server_offline():
    """When no models come back, guide the user instead of failing silently."""
    snippet = _ollama_branch_snippet()
    assert "ollama serve" in snippet, snippet[:600]
    assert "ollama pull" in snippet, snippet[:600]


def test_ollama_branch_is_distinct_from_ollama_cloud():
    """Local and cloud Ollama must be handled by separate branches."""
    src = inspect.getsource(flows_mod)
    assert 'provider_id == "ollama"' in src
    assert 'provider_id == "ollama-cloud"' in src
    # The bare-ollama branch must not accidentally call the cloud fetcher.
    snippet = _ollama_branch_snippet()
    assert "fetch_ollama_cloud_models" not in snippet, snippet[:400]
