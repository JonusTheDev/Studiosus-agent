"""The Governor — keep a local Ollama model inside the one GPU's power band.

Ported from the Studiosus (LuceRenascimur) ``core/governor.py``. When a local
model plus its growing KV cache no longer fits VRAM, Ollama does not refuse the
work — it quietly moves layers to system RAM and carries on, slower. Nothing
*looks* wrong; the answers still come. That silent spill is the cliff, and this
module's whole purpose is to keep us off it on purpose rather than by luck — and
above all, to never let a spill happen *silently*.

Two layers, exactly as in the original:

  * **Senses** (live I/O): what the card reports (``nvidia-smi``) and what Ollama
    holds resident right now. Ollama's ``/api/ps`` gives each resident model a
    ``size`` and a ``size_vram`` — their ratio IS the spill, read from the
    machine itself.
  * **Judgment** (pure, no GPU): given a model's measured power curve (built by
    the dyno bench, phase 2), choose the largest context that holds the band;
    cap a request that would spill, out loud; refuse a run whose model is not
    even installed — because through an error-swallowing client an absent model
    becomes empty output that looks like the model "said nothing".

Carved lesson (Studiosus docs/engine_modulator.md §6): one box can run more than
one Ollama server on the same GPU, each answering ``localhost:11434`` in its own
world. So every sense takes the endpoint explicitly, and every cached measurement
is keyed ``model@base_url`` — "not installed" is a claim about one server, never
about the box.

Scope: **local Ollama endpoints only.** Cloud / provider-agnostic paths never
enter here (the caller gates on ``is_local_endpoint`` + Ollama server type).
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# --- the band ------------------------------------------------------------------
# A dense 24B model can never sit fully inside 16 GB, so the band is a floor on
# the resident fraction, not a demand for 100%: hold at or above this and the
# spill stays on the gentle slope of the curve.
MIN_GPU_FRAC = 0.90

# Below the band, depth is free when speed is equal: a point within this fraction
# of the fastest measured tok/s counts as "just as fast", and the deepest such
# context wins. (The MoE lesson: off-band, tok/s is the truth, not gpu_frac.)
SPEED_TOLERANCE = 0.95

# nvidia-smi query fields, in the order we parse them back out.
_SMI_FIELDS = ("name", "memory.total", "memory.used", "memory.free",
               "temperature.gpu", "power.draw", "utilization.gpu")


def _default_log(tag: str, msg: str) -> None:
    logger.info("governor[%s]: %s", tag, msg)


LogFn = Callable[[str, str], None]


# --- senses: reading the real machine ------------------------------------------
# Ollama's REST endpoints (/api/ps, /api/tags, /api/version) live at the server
# root, not under the OpenAI-compat /v1 prefix. We reuse Hermes's own helpers so
# the host/auth handling matches the rest of the codebase exactly.

def _ollama_root(base_url: str) -> str:
    """The Ollama server root for a base_url, with any /v1 suffix stripped."""
    from agent.model_metadata import _localhost_to_ipv4
    root = _localhost_to_ipv4((base_url or "").rstrip("/"))
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


def _ollama_get(base_url: str, path: str, api_key: str = "",
                timeout: float = 3.0) -> Optional[dict]:
    """GET an Ollama JSON endpoint; None on any error (unreachable, non-200)."""
    import httpx
    from agent.model_metadata import _auth_headers
    try:
        with httpx.Client(timeout=timeout, headers=_auth_headers(api_key)) as client:
            resp = client.get(f"{_ollama_root(base_url)}{path}")
            if resp.status_code != 200:
                return None
            return resp.json()
    except Exception:
        return None


def _nvidia_smi() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(_SMI_FIELDS),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# read_gpu() shells out to nvidia-smi; agent init runs once per gateway message,
# so cache the reading briefly. VRAM total is static; free VRAM drifts slowly
# relative to a 20s window, and the governor only needs a coarse picture.
_gpu_cache: dict[str, Any] = {"value": None, "ts": 0.0}
_GPU_TTL = 20.0


def read_gpu(runner: Optional[Callable[[], str]] = None,
             use_cache: bool = True) -> Optional[dict]:
    """The gauge cluster: name, VRAM total/used/free (MB), temp, power, util.

    Returns None when nvidia-smi is absent (a CPU-only box); the governor then
    steers by Ollama's own split alone (or no-ops). ``runner`` is injectable for
    tests.
    """
    if use_cache and runner is None and (time.time() - _gpu_cache["ts"]) < _GPU_TTL:
        return _gpu_cache["value"]
    raw = (runner or _nvidia_smi)()
    result: Optional[dict] = None
    if raw:
        parts = [p.strip() for p in raw.strip().splitlines()[0].split(",")]
        if len(parts) >= len(_SMI_FIELDS):
            def num(x: str) -> Optional[float]:
                try:
                    return float(x)
                except ValueError:
                    return None
            result = {
                "name": parts[0],
                "vram_total_mb": num(parts[1]),
                "vram_used_mb": num(parts[2]),
                "vram_free_mb": num(parts[3]),
                "temp_c": num(parts[4]),
                "power_w": num(parts[5]),
                "util_pct": num(parts[6]),
            }
    if runner is None:
        _gpu_cache["value"], _gpu_cache["ts"] = result, time.time()
    return result


def gpu_fraction(size: Optional[float], size_vram: Optional[float]) -> float:
    """What fraction of a resident model sits on the GPU. 1.0 = all of it."""
    if not size:
        return 0.0
    return max(0.0, min(1.0, (size_vram or 0) / size))


def loaded_split(base_url: str, api_key: str = "") -> list[dict]:
    """Every model this Ollama holds resident right now, with its GPU/RAM split."""
    data = _ollama_get(base_url, "/api/ps", api_key)
    if not data:
        return []
    out = []
    for m in data.get("models", []):
        size = m.get("size", 0)
        vram = m.get("size_vram", 0)
        out.append({
            "model": m.get("name") or m.get("model"),
            "resident_gb": size / (1024 ** 3),
            "gpu_frac": gpu_fraction(size, vram),
            "offload_gb": max(0, size - vram) / (1024 ** 3),
        })
    return out


def model_split(base_url: str, model: str, api_key: str = "") -> tuple[Optional[int], Optional[int]]:
    """(size_bytes, size_vram_bytes) for one resident model, or (None, None)."""
    data = _ollama_get(base_url, "/api/ps", api_key)
    if data:
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if name == model or _bare_tag(name) == _bare_tag(model):
                return m.get("size", 0), m.get("size_vram", 0)
    return None, None


# installed_models() is consulted on every agent init; cache per-endpoint so a
# per-message gateway rebuild issues at most one /api/tags call per TTL window.
_tags_cache: dict[str, Any] = {}
_TAGS_TTL = 60.0


def installed_models(base_url: str, api_key: str = "",
                     use_cache: bool = True) -> Optional[list[str]]:
    """Model tags THIS Ollama server has pulled. None if it was unreachable.

    The None-vs-[] distinction is load-bearing: None = "server unreachable, do
    not judge"; [] = "server has nothing". Judge a model absent only from the
    host the harness truly uses (the carved multi-server lesson).
    """
    root = _ollama_root(base_url)
    if use_cache:
        hit = _tags_cache.get(root)
        if hit is not None and (time.time() - hit[1]) < _TAGS_TTL:
            return hit[0]
    data = _ollama_get(base_url, "/api/tags", api_key)
    if data is None:
        return None  # unreachable — never memoize a failure
    tags = [m.get("name") or m.get("model") for m in data.get("models", [])]
    _tags_cache[root] = (tags, time.time())
    return tags


def server_version(base_url: str, api_key: str = "") -> Optional[str]:
    """This Ollama server's version string, or None if it was unreachable."""
    data = _ollama_get(base_url, "/api/version", api_key)
    return data.get("version") if data else None


# --- judgment: pure reasoning over a measured power curve -----------------------
# A power curve is the dyno's record for one model: a list of measured points,
#   {"num_ctx": int, "resident_gb": float, "gpu_frac": float, "tok_s": float}
# stored under the model's cached "power" block.

def recommend_num_ctx(curve: list[dict], min_gpu_frac: float = MIN_GPU_FRAC,
                      ceiling: Optional[int] = None) -> Optional[dict]:
    """The largest measured context whose resident fraction holds the band.

    When no point holds the band (a model too large for any useful context), the
    band has failed as a guide — and gpu_frac can lie off-band (a forced split
    wore 96% "residence" at a fifth of the speed). So below the band we steer by
    measured truth: among points within SPEED_TOLERANCE of the fastest, take the
    deepest context — speed first, depth when it is speed-free. Callers must
    still check the returned point's gpu_frac against the floor. Returns a curve
    point, or None for an empty curve.
    """
    if not curve:
        return None
    in_band = [p for p in curve if p["gpu_frac"] >= min_gpu_frac
               and (ceiling is None or p["num_ctx"] <= ceiling)]
    if in_band:
        return max(in_band, key=lambda p: (p["num_ctx"], p.get("tok_s", 0)))
    eligible = [p for p in curve if ceiling is None or p["num_ctx"] <= ceiling] or curve
    fastest = max(p.get("tok_s", 0) for p in eligible)
    near = [p for p in eligible if p.get("tok_s", 0) >= fastest * SPEED_TOLERANCE]
    return max(near, key=lambda p: (p["num_ctx"], p.get("tok_s", 0)))


def _nearest(curve: list[dict], num_ctx: int) -> dict:
    return min(curve, key=lambda p: abs(p["num_ctx"] - num_ctx))


def check_plan(profile: dict, num_ctx: int,
               min_gpu_frac: float = MIN_GPU_FRAC) -> dict:
    """Would this profile at this context hold the band? Judged from its curve.

    Returns {"status": "in_band"|"spills"|"unknown", "gpu_frac", "tok_s",
    "recommended_num_ctx", "note"}. "unknown" means the model was never put on
    the dyno — we are flying blind, and we say so rather than guess.
    """
    power = profile.get("power")
    if not power or not power.get("curve"):
        return {"status": "unknown", "gpu_frac": None, "tok_s": None,
                "recommended_num_ctx": None,
                "note": "no power curve; run the dyno before trusting the band"}
    curve = power["curve"]
    point = _nearest(curve, num_ctx)
    rec = recommend_num_ctx(curve, min_gpu_frac)
    status = "in_band" if point["gpu_frac"] >= min_gpu_frac else "spills"
    return {"status": status, "gpu_frac": point["gpu_frac"], "tok_s": point.get("tok_s"),
            "recommended_num_ctx": rec["num_ctx"] if rec else None,
            "note": "judged from nearest measured point num_ctx=%d" % point["num_ctx"]}


def govern_options(profile: dict, options: dict,
                   log_fn: LogFn = _default_log) -> dict:
    """Cap a run's options so the engine holds its band — always out loud.

    A requested num_ctx above the profile's dyno recommendation (or an unset one)
    is lowered to it, and the cap is logged: we never spill silently, and we
    never cap silently either. A profile with no curve runs at the requested
    context but is flagged BLIND. A ``num_ctx_decree`` (the farmer's chosen
    operating context) overrides the counsel — and when it does, the price is
    said out loud. Never mutates the input.
    """
    opts = dict(options)
    power = profile.get("power") or {}
    rec = power.get("recommended_num_ctx")
    decree = profile.get("num_ctx_decree")
    name = profile.get("name", profile.get("model", "?"))
    req = opts.get("num_ctx")
    if decree:
        if rec and decree > rec:
            log_fn("governor", "%s: the decree holds num_ctx=%d over the dyno's "
                   "counsel of %d — depth chosen with open eyes" % (name, decree, rec))
        if req is None or req > decree:
            log_fn("governor", "%s: capping num_ctx %s -> %d (the decree)"
                   % (name, req, decree))
            opts["num_ctx"] = decree
    elif rec:
        if req is None or req > rec:
            log_fn("governor", "%s: capping num_ctx %s -> %d to hold the power band"
                   % (name, req, rec))
            opts["num_ctx"] = rec
    else:
        log_fn("governor", "%s: no power curve — running num_ctx=%s BLIND "
               "(run: hermes dyno %s)" % (name, req, name))
    rec_ng = power.get("recommended_num_gpu")
    if rec_ng is not None and opts.get("num_gpu") is None:
        log_fn("governor", "%s: pinning num_gpu=%d (the dyno's winning layer split)"
               % (name, rec_ng))
        opts["num_gpu"] = rec_ng
    return opts


def preflight(profile: dict, installed: Optional[list[str]],
              gpu: Optional[dict] = None,
              min_gpu_frac: float = MIN_GPU_FRAC) -> dict:
    """Refuse a run that cannot succeed; warn about one that will limp.

    Pure over facts already gathered (``installed`` = this server's tags, or None
    if it was unreachable; ``gpu`` = read_gpu() or None). The load-bearing guard:
    a configured model absent from the server would surface as silent empty
    output downstream — caught here it becomes a loud, plain refusal.
    """
    model = profile["model"]
    warnings: list[str] = []
    if installed is not None and not _model_installed(model, installed):
        return {"ok": False,
                "reason": "model %r is not installed on this Ollama server (it has: %s)"
                          % (model, ", ".join(sorted(installed)) or "nothing"),
                "warnings": warnings}
    if not profile.get("power"):
        warnings.append("no power curve for %s — the band is a guess until the "
                        "dyno has run" % profile.get("name", model))
    else:
        req = (profile.get("options") or {}).get("num_ctx")
        if req is not None:
            plan = check_plan(profile, req, min_gpu_frac)
            if plan["status"] == "spills":
                warnings.append(
                    "num_ctx=%s would spill below the band (%.0f%% on GPU); the "
                    "governor will cap it to %s"
                    % (req, (plan["gpu_frac"] or 0) * 100, plan["recommended_num_ctx"]))
    return {"ok": True, "reason": "ready", "warnings": warnings}


# --- model-tag matching --------------------------------------------------------

def _bare_tag(model: Optional[str]) -> str:
    """Strip Hermes's provider prefix from a model name (e.g. ``ollama/x`` -> ``x``)."""
    if not model:
        return ""
    try:
        from agent.model_metadata import _strip_provider_prefix
        return _strip_provider_prefix(model)
    except Exception:
        return model.split("/", 1)[-1] if "/" in model else model


def _model_installed(model: str, installed: list[str]) -> bool:
    """Is ``model`` present among an Ollama server's tags? Generous on tags.

    Ollama tags carry a ``:tag`` suffix (defaulting to ``:latest``) that a
    config model name may omit, so we match exactly, then on ``:latest``, then
    on the base name before ``:`` — a lenient match to avoid falsely refusing a
    model that is really installed under a fuller tag.
    """
    bare = _bare_tag(model).lower()
    if not bare:
        return True  # nothing to check against — do not refuse
    tags = {(t or "").lower() for t in installed}
    if bare in tags or f"{bare}:latest" in tags:
        return True
    if bare.endswith(":latest") and bare[:-len(":latest")] in tags:
        return True
    base = bare.split(":", 1)[0]
    return any((t.split(":", 1)[0] == base) for t in tags)


# --- power-curve cache ---------------------------------------------------------
# Reuses Hermes's cache convention: a single YAML under the hermes home, keyed
# ``model@base_url`` (the dyno command in phase 2 writes curves here).

def _curve_cache_path():
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "governor_curves.yaml"


def _curve_key(model: str, base_url: str) -> str:
    return f"{_bare_tag(model)}@{(base_url or '').rstrip('/')}"


def load_curve(model: str, base_url: str) -> Optional[dict]:
    """The persisted ``power`` block for a model@endpoint, or None."""
    path = _curve_cache_path()
    if not path.exists():
        return None
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return (data.get("curves") or {}).get(_curve_key(model, base_url))
    except Exception as exc:
        logger.debug("governor: failed to load curve cache: %s", exc)
        return None


def save_curve(model: str, base_url: str, power: dict) -> None:
    """Persist a measured ``power`` block for a model@endpoint (phase-2 dyno)."""
    path = _curve_cache_path()
    try:
        import yaml
        data = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        curves = data.get("curves") or {}
        curves[_curve_key(model, base_url)] = power
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"curves": curves}, f, default_flow_style=False)
    except Exception as exc:
        logger.debug("governor: failed to save curve cache: %s", exc)


# --- orchestration: the phase-1 pre-flight the agent seam calls -----------------

@dataclass
class GovernResult:
    """Outcome of a governor pre-flight for a local Ollama request.

    ``safe_ctx`` is the num_ctx to actually request (possibly capped). ``warnings``
    are user-facing lines to surface out loud. ``refusal`` is set (and safe_ctx
    None) only when the run cannot succeed — the caller raises on it.
    """
    safe_ctx: Optional[int]
    warnings: list[str] = field(default_factory=list)
    refusal: Optional[str] = None


def resolve_local_num_ctx(
    *,
    model: str,
    base_url: str,
    api_key: str = "",
    requested_ctx: Optional[int],
    ceiling: Optional[int] = None,
    decree: Optional[int] = None,
    min_gpu_frac: float = MIN_GPU_FRAC,
    tool_use_floor: int = 64_000,
    logger_obj: Optional[logging.Logger] = None,
) -> GovernResult:
    """Phase-1 governor for a local Ollama endpoint. Cheap, always-on, honest.

    Order of operations:
      1. Gate on Ollama server type (cached). Non-Ollama → no-op passthrough.
      2. Refuse-if-absent: a model missing from THIS server's tags → loud refusal
         (upgrades today's silent empty-output failure).
      3. Cap by a cached power curve if present, honoring the decree; then apply
         the user's ``ceiling`` (model.context_length), out loud.
      4. Measured spill guard (no curve needed): if the model is already resident
         and spilling to RAM, warn loudly.
      5. Tool-use floor: never silently cap below Hermes's 64K floor — keep the
         floor and warn about the spill instead.

    Never raises for spill/measurement; only reports. The caller decides what to
    do with ``refusal``. Callers should wrap this in try/except so a governor bug
    can never break a working setup (fail-open).
    """
    log = logger_obj or logger
    warnings: list[str] = []

    # 1. Ollama-only gate (detect_local_server_type is process-cached).
    try:
        from agent.model_metadata import detect_local_server_type
        if detect_local_server_type(base_url, api_key=api_key) != "ollama":
            return GovernResult(requested_ctx, warnings, None)
    except Exception:
        return GovernResult(requested_ctx, warnings, None)

    bare = _bare_tag(model)

    # 2. Refuse-if-absent (loud, not silent-empty).
    installed = installed_models(base_url, api_key=api_key)
    if installed is not None and not _model_installed(model, installed):
        return GovernResult(
            None, warnings,
            "model %r is not installed on this Ollama server (it has: %s)"
            % (model, ", ".join(sorted(installed)) or "nothing"),
        )

    curve = load_curve(model, base_url)
    profile: dict = {"model": bare, "name": model, "power": curve or {}}
    if decree:
        try:
            profile["num_ctx_decree"] = int(decree)
        except (TypeError, ValueError):
            pass

    # 3. Curve/decree cap (reuse the ported judgment; its logs go to debug — we
    #    synthesize the user-facing warning from the actual delta below).
    before = requested_ctx
    opts = govern_options(profile, {"num_ctx": requested_ctx},
                          log_fn=lambda tag, msg: log.debug("%s", msg))
    safe = opts.get("num_ctx", requested_ctx)
    if curve and before and safe and safe < before:
        warnings.append(
            "capped num_ctx %s → %s to hold the GPU's power band (measured)." % (before, safe))

    # 3b. Honor the user's explicit context_length ceiling, out loud.
    if ceiling and safe and safe > ceiling:
        warnings.append("capped num_ctx %s → %s (model.context_length)." % (safe, ceiling))
        safe = ceiling

    # 4. Measured spill guard — cheap, no curve required, only if already resident.
    size, size_vram = model_split(base_url, bare, api_key=api_key)
    if size:
        frac = gpu_fraction(size, size_vram)
        if frac < min_gpu_frac:
            spilled = max(0, size - (size_vram or 0)) / (1024 ** 3)
            warnings.append(
                "%s is spilling ~%.1f GB to system RAM (only %.0f%% resident) at "
                "num_ctx=%s — generations will be slow. Lower model.ollama_num_ctx, "
                "use a smaller/more-quantized model, or run `hermes dyno %s`."
                % (model, spilled, frac * 100, safe, model))

    # 5. Tool-use floor reconciliation — Hermes needs ≥64K for reliable tools, so
    #    never silently cap below it; keep the floor and be honest about the cost.
    if safe and safe < tool_use_floor:
        warnings.append(
            "the largest safe context for this GPU (%s) is below Hermes's %sK "
            "tool-use floor; keeping %s — expect a VRAM spill. Use a smaller or "
            "more-quantized model, or run `hermes dyno %s` and set a decree."
            % (safe, tool_use_floor // 1000, tool_use_floor, model))
        safe = tool_use_floor

    return GovernResult(safe, warnings, None)


def format_dashboard(gpu: Optional[dict], loaded: list[dict]) -> str:
    """The gauge cluster, printable. gpu = read_gpu(), loaded = loaded_split()."""
    lines = []
    if gpu:
        used = gpu.get("vram_used_mb") or 0
        total = gpu.get("vram_total_mb") or 0
        pct = (used / total * 100) if total else 0
        lines.append("GPU  %s" % gpu.get("name"))
        lines.append("VRAM %.0f / %.0f MB (%.0f%% used, %.0f MB free)"
                     % (used, total, pct, gpu.get("vram_free_mb") or 0))
    else:
        lines.append("GPU  (nvidia-smi unavailable)")
    if loaded:
        lines.append("resident models:")
        for m in loaded:
            band = "IN BAND" if m["gpu_frac"] >= MIN_GPU_FRAC else "SPILLING"
            lines.append("  %-28s %.2f GB  %.0f%% on GPU  (%.2f GB spilled)  %s"
                         % (m["model"], m["resident_gb"], m["gpu_frac"] * 100,
                            m["offload_gb"], band))
    else:
        lines.append("resident models: none")
    return "\n".join(lines)
