"""Ollama (local) provider profile.

First-class profile for a local Ollama server, reached through Ollama's
OpenAI-compatible ``/v1`` endpoint (the same transport as any OpenAI-style
provider — there is no separate native-Ollama transport to gain by
switching). The endpoint defaults to the standard local port so no
``base_url`` guesswork is needed; a running local server needs no API key.
Use the ``ollama-cloud`` profile for the hosted service.

Transport quirks (kept in sync with the ``custom`` profile, which also
serves local Ollama's siblings vLLM / llama.cpp / GLM-on-ARK):
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → top-level reasoning_effort="none"
    (Ollama /v1/chat/completions ignores think=False — ollama#14820)
    + extra_body.think = False for /api/chat and proxies
  - a generous default_max_tokens floor so responses aren't truncated at
    Ollama's internal num_predict=128 default (#39281).
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


class OllamaProfile(ProviderProfile):
    """Local Ollama — think=false, num_ctx, and a baked-in default endpoint."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Reasoning / thinking control. Disabled → stop a thinking-capable
        # model from reasoning; Ollama's /v1/chat/completions silently ignores
        # extra_body.think (only /api/chat honours it — ollama#14820) but
        # respects the top-level reasoning_effort field, so both are needed
        # (#25758). We deliberately do NOT emit think=True on enable: thinking
        # is server-default-on, and forcing the Ollama-only flag risks a 400
        # on proxies that don't recognize it.
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                top_level["reasoning_effort"] = "none"
                extra_body["think"] = False
            elif _effort:
                top_level["reasoning_effort"] = _effort

        return extra_body, top_level


ollama = OllamaProfile(
    name="ollama",
    display_name="Ollama (local)",
    description="Ollama (Local model server, default http://127.0.0.1:11434)",
    env_vars=("OLLAMA_API_KEY", "OLLAMA_BASE_URL"),
    base_url=DEFAULT_OLLAMA_BASE_URL,
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). Only a
    # floor used when the user hasn't set model.max_tokens.
    default_max_tokens=65536,
)

register_provider(ollama)
