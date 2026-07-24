"""The model-details report — what the desktop's Dyno rail renders.

Joins, for every model this local Ollama server has pulled, four honest
sources into one payload:

* the **Dyno bench** (``agent/dyno.py`` profiles) — synthetic throughput across
  context sizes, the recommended/operating context, GPU residence;
* the **live history** (``agent/dyno_history.py``) — real observed throughput
  from actual turns, so the bench figure can be checked against reality;
* **specialties** — auto-derived from the server's own ``/api/show``
  capabilities (tools / vision / reasoning / …), never invented;
* the **running process** (``agent/flame.py``) — loaded / pinned / on-GPU / ready.

Read-only and honest: an unreachable server yields an empty model list (not an
error), a model never benched says so plainly (``benched: false`` with null
KPIs), and a model with no live samples reports ``live: null`` — no data is not
zero.  Only local models appear; Dyno throughput is a local-GPU truth and a
remote API model has none.
"""

from __future__ import annotations

from typing import Optional

from agent import dyno, dyno_history, flame


# Ollama /api/show `capabilities` entries -> the specialty badge we show.
# "completion" is the base generation capability every model has; it earns no
# badge. Anything unmapped is passed through lowercased so a new Ollama
# capability still surfaces rather than being silently dropped.
_CAP_TO_SPECIALTY = {
    "tools": "tools",
    "vision": "vision",
    "thinking": "reasoning",
    "reasoning": "reasoning",
    "insert": "fill-in",
    "embedding": "embedding",
}


def _tok_s_at(profile: Optional[dict], num_ctx: int) -> Optional[float]:
    """Generation throughput at an exact context point, or None if not measured."""
    curve = ((profile or {}).get("power") or {}).get("curve") or []
    for point in curve:
        if point.get("num_ctx") == num_ctx:
            return point.get("tok_s")
    return None


def _kpi(profile: Optional[dict]) -> dict:
    """The headline KPI columns, derived from a Dyno profile."""
    power = (profile or {}).get("power") or {}
    return {
        "tok_s_at_64k": _tok_s_at(profile, 65536),
        "tok_s_at_32k": _tok_s_at(profile, 32768),
        "recommended_num_ctx": power.get("recommended_num_ctx"),
        "recommended_tok_s": power.get("recommended_tok_s"),
        "recommended_in_band": power.get("recommended_in_band"),
        "fully_fits": power.get("fully_fits"),
    }


def _specialties(model: str, host: str) -> list[str]:
    """Auto-derived specialty badges from the server's own /api/show.

    Honest and local: reads the model's declared ``capabilities`` (Ollama
    0.6.0+). Best-effort — an unreachable/older server yields an empty list
    rather than a guess.
    """
    try:
        data = dyno._http(host.rstrip("/") + "/api/show",
                          {"name": model}, timeout=10)
    except Exception:
        return []
    caps = data.get("capabilities")
    if not isinstance(caps, list):
        return []
    out: list[str] = []
    for cap in caps:
        key = str(cap).lower()
        if key == "completion":
            continue
        badge = _CAP_TO_SPECIALTY.get(key, key)
        if badge not in out:
            out.append(badge)
    return out


def _status(inspection: dict) -> dict:
    """The compact running-process fields the rail's per-row status shows."""
    return {
        "loaded": inspection.get("loaded", False),
        "pinned": inspection.get("pinned", False),
        "gpu_frac": inspection.get("gpu_frac"),
        "loaded_num_ctx": inspection.get("loaded_num_ctx"),
        "ready": inspection.get("ready", False),
    }


def build_model_report(host: Optional[str] = None,
                       current_model: Optional[str] = None) -> dict:
    """One payload for the desktop model-details rail.

    ``current_model`` (the session's active model, if known) drives the
    running-process header. Never raises: an unreachable host returns an empty
    model list and ``flame: null``.
    """
    host = host or dyno.default_host()
    installed = dyno.installed_models(host)
    if installed is None:  # server unreachable — say so plainly, don't error
        return {"host": host, "reachable": False, "models": [], "flame": None}

    # Running-process header: inspect the active model when we know it; when it
    # is unknown the header simply hides (the per-row status still shows what is
    # resident, so nothing is lost).
    flame_report = None
    try:
        if current_model:
            flame_report = flame.inspect(current_model, host=host)
    except Exception:
        flame_report = None

    models = []
    for model in installed:
        if not model:
            continue
        profile = dyno.load_profile(model)
        try:
            inspection = flame.inspect(model, host=host)
        except Exception:
            inspection = {}
        models.append({
            "model": model,
            "benched": profile is not None,
            "operating_num_ctx": dyno.operating_num_ctx(profile),
            "kpi": _kpi(profile),
            "profile": profile.get("power") if profile else None,
            "chosen_because": (profile or {}).get("chosen_because"),
            "specialties": _specialties(model, host),
            "live": dyno_history.summary(model),
            "status": _status(inspection),
        })

    # Benched models first (they have KPIs to compare), then alphabetical.
    models.sort(key=lambda m: (not m["benched"], m["model"]))
    return {
        "host": host,
        "reachable": True,
        "current_model": current_model,
        "flame": flame_report,
        "models": models,
    }
