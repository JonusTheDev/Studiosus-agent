"""The dyno bench — measure a local model's real power curve, don't assume it.

Ported from the Studiosus (LuceRenascimur) ``studiosus/dyno.py``. We put a local
Ollama model on the rollers and sweep it across context sizes. At each point we
read two truths straight from the machine: how much of the model stayed on the
GPU (Ollama's own ``size_vram / size``) and how fast it truly generated
(``eval_count / eval_duration`` — tok/s from Ollama's timings, not a stopwatch).
The measured points are the model's **power curve**; the Governor (agent/governor.py)
later picks the operating context from it — the largest that holds the band.

Two covenants, both learned the hard way in the original:

  * The power block records the **endpoint** it was measured against. One box can
    run more than one Ollama on the same GPU; a curve from the wrong server is a
    seed for a later fracture. Data that does not name its server is not data.
  * The recommendation carries ``recommended_in_band``. When no context holds the
    band, the least-spilling point is still written down — but flagged false, so
    a model that never fits can never wear its fallback as a fitness.

The sweep logic is pure and mock-tested; the live roller is injected. Phase-2
adaptations for Hermes: no Studiosus profile JSON (the curve is persisted through
``governor.save_curve`` / read back by ``governor.load_curve``), no vision-probe
staging, and a sweep that reaches into Hermes's ≥64K tool-use range.

Run live:  ``hermes dyno <model>``
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from agent import governor

DYNO_VERSION = 1

# The sweep: powers of two across the useful range, reaching past Hermes's 64K
# tool-use floor so the recommendation can actually land in tool-capable
# territory on a GPU that can hold it. A point the model cannot hold (Ollama
# refuses a num_ctx beyond its trained window, or errors) is simply skipped.
DEFAULT_CTX_POINTS = (4096, 8192, 16384, 32768, 65536, 131072)

# A fixed probe: enough prompt to seed the KV cache, enough output to time honestly.
PROBE_PROMPT = (
    "You are reasoning about a software design. In two or three short paragraphs, "
    "explain the tradeoff between running a large language model fully on the GPU "
    "versus offloading some layers to system RAM. Be concrete about throughput.")
PROBE_TOKENS = 120


def run_dyno(
    model: str,
    measure_fn: Callable[..., Optional[dict]],
    *,
    base_url: str = "",
    ctx_points=DEFAULT_CTX_POINTS,
    gpu: Optional[dict] = None,
    min_gpu_frac: float = governor.MIN_GPU_FRAC,
    write: bool = True,
    server_env: Optional[dict] = None,
    num_gpu_points=None,
    repeats: int = 1,
) -> dict:
    """Sweep one model, build its power curve, and (optionally) persist it.

    ``measure_fn(model, num_ctx)`` returns
    ``{"resident_gb", "gpu_frac", "tok_s", "prompt_tok_s", "load_s"}`` or None to
    skip a point the model cannot run. ``num_gpu_points`` crosses each context
    with explicit layer splits (measure_fn then gets a ``num_gpu`` keyword).
    ``repeats`` rolls each point that many times and records the **mean** of every
    metric — a single probe wanders run-to-run, wider than the differences the
    recommendation must judge. Points that error are dropped from the mean; a
    point that never succeeds is skipped.

    Returns the ``power`` block; with ``write`` it is saved via
    ``governor.save_curve(model, base_url, power)`` for the Governor to read.
    """
    curve: list[dict] = []
    for nc in ctx_points:
        for ng in (num_gpu_points or (None,)):
            kwargs: dict[str, Any] = {}
            if ng is not None:
                kwargs["num_gpu"] = ng
            rolls = []
            for _ in range(max(1, repeats)):
                m = measure_fn(model, nc, **kwargs) if kwargs else measure_fn(model, nc)
                if m is not None:
                    rolls.append(m)
            if not rolls:
                continue

            def mean(key: str, default: float = 0.0) -> float:
                return sum(r.get(key, default) for r in rolls) / len(rolls)

            point = {"num_ctx": nc,
                     "resident_gb": round(mean("resident_gb"), 2),
                     "gpu_frac": round(mean("gpu_frac"), 3),
                     "tok_s": round(mean("tok_s"), 1),
                     "prompt_tok_s": round(mean("prompt_tok_s"), 1),
                     "load_s": round(mean("load_s"), 1)}
            if repeats > 1:
                point["rolls"] = len(rolls)
            if ng is not None:
                point["num_gpu"] = ng
            curve.append(point)

    rec = governor.recommend_num_ctx(curve, min_gpu_frac)
    power = {
        "dyno_version": DYNO_VERSION,
        "date": time.strftime("%Y-%m-%d"),
        "host": (base_url or "").rstrip("/"),
        "gpu": gpu,
        "server_env": server_env,
        "min_gpu_frac": min_gpu_frac,
        "curve": curve,
        "recommended_num_ctx": rec["num_ctx"] if rec else None,
        "recommended_num_gpu": rec.get("num_gpu") if rec else None,
        "recommended_tok_s": rec["tok_s"] if rec else None,
        "recommended_in_band": bool(rec) and rec["gpu_frac"] >= min_gpu_frac,
        "fully_fits": bool(rec) and rec["gpu_frac"] >= 0.999,
    }
    if write:
        governor.save_curve(model, base_url, power)
    return power


def live_measure(base_url: str, api_key: str = "") -> Callable[..., Optional[dict]]:
    """The real roller: clear the field, run one timed generation, read the split.

    Returns a ``measure_fn`` for :func:`run_dyno`. A generation that errors
    (context too large for the GPU, model refusing) yields None, so the sweep
    skips the point instead of recording a false zero.
    """
    def measure(model: str, num_ctx: int, num_gpu: Optional[int] = None,
                images: Optional[list] = None) -> Optional[dict]:
        governor.unload_all(base_url, api_key=api_key)
        time.sleep(1)
        resp = governor.generate_raw(
            base_url, model, PROBE_PROMPT, num_ctx,
            num_predict=PROBE_TOKENS, num_gpu=num_gpu, images=images, api_key=api_key)
        if not resp:
            return None
        size, vram = governor.model_split(base_url, model, api_key=api_key)
        if not size:
            return None
        eval_count = resp.get("eval_count", 0)
        eval_dur = resp.get("eval_duration", 0) / 1e9
        p_count = resp.get("prompt_eval_count", 0)
        p_dur = resp.get("prompt_eval_duration", 0) / 1e9
        return {
            "resident_gb": size / (1024 ** 3),
            "gpu_frac": governor.gpu_fraction(size, vram),
            "tok_s": (eval_count / eval_dur) if eval_dur else 0.0,
            "prompt_tok_s": (p_count / p_dur) if p_dur else 0.0,
            "load_s": resp.get("load_duration", 0) / 1e9,
        }
    return measure


def format_power(power: dict, model: str) -> str:
    """A readable summary of a measured power block for the CLI."""
    lines = [f"Power curve for {model}  (measured {power.get('date', '?')})"]
    host = power.get("host")
    if host:
        lines.append(f"  endpoint: {host}")
    gpu = power.get("gpu") or {}
    if gpu.get("name"):
        lines.append("  GPU: %s (%.0f MB VRAM)" % (gpu.get("name"), gpu.get("vram_total_mb") or 0))
    lines.append("")
    lines.append("  %-9s  %-8s  %-8s  %-9s  %s" % ("num_ctx", "on GPU", "tok/s", "prefill", "resident"))
    for p in power.get("curve", []):
        band = "in band" if p["gpu_frac"] >= power.get("min_gpu_frac", governor.MIN_GPU_FRAC) else "SPILLING"
        lines.append("  %-9d  %6.0f%%  %8.1f  %9.1f  %5.1f GB  %s"
                     % (p["num_ctx"], p["gpu_frac"] * 100, p.get("tok_s", 0),
                        p.get("prompt_tok_s", 0), p.get("resident_gb", 0), band))
    lines.append("")
    rec = power.get("recommended_num_ctx")
    if rec:
        flag = "" if power.get("recommended_in_band") else "  (nothing holds the band — least-spilling fallback)"
        lines.append("  → recommended num_ctx: %d  (%.1f tok/s)%s"
                     % (rec, power.get("recommended_tok_s") or 0, flag))
    else:
        lines.append("  → no usable measurements (is the server running and the model pulled?)")
    return "\n".join(lines)
