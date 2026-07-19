"""The dyno bench — where a model's strength is measured, not assumed.

Hermes can be pointed at a local Ollama, but nothing in it answers the
question that actually decides whether a local setup is pleasant to use:
*where does this model make power without spilling off the GPU?*  Context
length is usually guessed, copied from a friend's config, or set to whatever
the model card claims.  On a 16 GB card running a 25 GB MoE, a guess is just a
slower answer arriving later.

So we put the model on the rollers and sweep it across context sizes.  At each
point we read two truths straight from the machine: how much of the model
stayed on the GPU (Ollama's own ``size_vram / size``) and how fast it truly
generated (``eval_count / eval_duration`` — from Ollama's timings, not a
stopwatch).  The measured points are the model's **power curve**, and
:func:`recommend_num_ctx` picks the operating context from it.

Distilled from the ancestor Loom's ``studiosus/dyno.py`` + ``core/governor.py``
(see ``.plans/studiosus-heart-strategy.md``), rebuilt on Hermes' bricks.  Three
covenants came across, each learned the hard way:

  * **Data that does not name its server is not data.**  A curve records the
    host, the GPU, and the server environment it was measured under.  One box
    can run more than one Ollama; a curve from the wrong server is a seed for
    a later fracture.
  * **A model that never fits may not wear its fallback as a fitness.**  When
    no context holds the GPU band, the least-bad point is still recorded — but
    flagged ``recommended_in_band: false``.
  * **Off-band, gpu_frac lies.**  The ancestor measured a forced split wearing
    96% "residence" at a fifth of the speed.  Below the band we therefore steer
    by measured tok/s, not by residence.  See :func:`recommend_num_ctx`.

The bench logic is pure and mock-tested; the live roller is injected, so the
whole judgment path runs in tests without a GPU.

Run live against this box's Ollama::

    python -m agent.dyno --model qwen3.6:latest
    python -m agent.dyno --list
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from hermes_constants import get_hermes_home

DYNO_VERSION = 1

# The GPU band.  A model holding at least this fraction of itself in VRAM is
# "in band"; below it, the machine is paging weights across PCIe every token.
MIN_GPU_FRAC = 0.90

# Off-band tie-break: among points within this fraction of the fastest measured
# speed, prefer the deepest context.  Depth is worth having whenever it is free.
SPEED_TOLERANCE = 0.95

# The sweep.  Starts at 32k because that is the floor Hermes asks of a model,
# and climbs to 128k for the models that can genuinely hold it.  A point the
# model cannot run (Ollama refuses a num_ctx beyond its trained window) is
# skipped rather than recorded as a false zero.
DEFAULT_CTX_POINTS = (32768, 65536, 131072)

# A fixed probe: enough prompt to seed the KV cache, enough output to time
# honestly.  Identical across models, or the curves are not comparable.
PROBE_PROMPT = (
    "You are reasoning about a software design. In two or three short paragraphs, "
    "explain the tradeoff between running a large language model fully on the GPU "
    "versus offloading some layers to system RAM. Be concrete about throughput."
)
PROBE_TOKENS = 120

# Shipped measurements live in the repo (tracked — a measurement is distilled
# data and must not be lost to a cleared cache).  A user's own measurements
# land in their Hermes home and always win: our numbers describe our box, and
# a stranger's 8 GB laptop deserves its own truth rather than ours.
SEEDED_PROFILE_DIR = Path(__file__).resolve().parent.parent / "dyno_profiles"


def measured_profile_dir() -> Path:
    """Where this user's own dyno runs are recorded."""
    return get_hermes_home() / "dyno"


# --- talking to Ollama ---------------------------------------------------


DEFAULT_PORT = 11434

# Addresses that mean "bind everywhere" and are useless to dial.  Ollama's
# OLLAMA_HOST serves double duty — the server reads it as a bind address, the
# client as a dial address — so a box configured to listen on all interfaces
# exports OLLAMA_HOST=0.0.0.0 into every shell, and a client that trusts it
# blindly ends up asking http://0.0.0.0 for a model list.  Measured on this
# box, 2026-07-18.
_WILDCARD_HOSTS = {"0.0.0.0", "::", "[::]", "*"}


def default_host() -> str:
    """The Ollama this box actually uses, as something dialable.

    ``OLLAMA_HOST`` is honored — it is how a WSL-hosted or remote server is
    usually named — but it is repaired rather than trusted: a bare
    ``host:port`` is given the scheme it omitted, a missing port is supplied,
    and a wildcard bind address is turned back into loopback.
    """
    raw = (os.environ.get("OLLAMA_HOST") or "").strip()
    if not raw:
        return f"http://localhost:{DEFAULT_PORT}"

    scheme = "http"
    if "://" in raw:
        scheme, _, raw = raw.partition("://")
    raw = raw.rstrip("/")

    host, _, port = raw.rpartition(":")
    if not host or not port.isdigit():  # no port given — the whole string is the host
        host, port = raw, str(DEFAULT_PORT)
    if host.lower() in _WILDCARD_HOSTS:
        host = "127.0.0.1"
    return f"{scheme}://{host}:{port}"


def server_env(host: Optional[str] = None) -> dict:
    """What this measurement was taken under: server version plus any knobs.

    Ollama exposes no endpoint for its own configuration, so the knobs are read
    from this process's environment.  That is honest but partial — a server
    started by systemd with its own ``Environment=`` lines will not show them
    here — so a caller who knows better should pass its own dict instead.
    """
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("OLLAMA_", "LLAMA_")) and k != "OLLAMA_HOST"}
    return {"ollama_version": server_version(host), "env": env}


def _http(url: str, payload: Optional[dict] = None, timeout: int = 1200) -> dict:
    """One JSON round-trip.  Raises on transport failure; callers decide."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_version(host: Optional[str] = None) -> Optional[str]:
    """The server's version, or None if unreachable.

    Recorded beside every curve: two measurements under different server
    builds are different experiments, and the block must say so.
    """
    try:
        return _http((host or default_host()) + "/api/version", timeout=10).get("version")
    except Exception:
        return None


def installed_models(host: Optional[str] = None) -> Optional[list[str]]:
    """Model tags THIS server has pulled, or None if it was unreachable.

    The inventory of one server, never of the box — judge a model absent only
    from the host the agent truly uses.
    """
    try:
        tags = _http((host or default_host()) + "/api/tags", timeout=30).get("models", [])
    except Exception:
        return None
    return [m.get("name") or m.get("model") for m in tags]


def gpu_fraction(size: float, size_vram: float) -> float:
    """What fraction of a resident model sits on the GPU.  1.0 = all of it."""
    if not size:
        return 0.0
    return max(0.0, min(1.0, size_vram / size))


def loaded_split(host: Optional[str] = None) -> list[dict]:
    """Every model this server holds resident right now, with its GPU/RAM split."""
    try:
        models = _http((host or default_host()) + "/api/ps", timeout=30).get("models", [])
    except Exception:
        return []
    out = []
    for m in models:
        size = m.get("size", 0)
        vram = m.get("size_vram", 0)
        out.append({
            "model": m.get("name") or m.get("model"),
            "resident_gb": size / (1024 ** 3),
            "gpu_frac": gpu_fraction(size, vram),
            "offload_gb": max(0, size - vram) / (1024 ** 3),
        })
    return out


def model_split(model: str, host: Optional[str] = None) -> tuple:
    """``(size_bytes, size_vram_bytes)`` for one resident model, or ``(None, None)``."""
    try:
        for m in _http((host or default_host()) + "/api/ps", timeout=30).get("models", []):
            if (m.get("name") or m.get("model")) == model:
                return m.get("size", 0), m.get("size_vram", 0)
    except Exception:
        pass
    return None, None


def clear_field(host: Optional[str] = None, keep: Iterable[str] = ()) -> list[str]:
    """Release every resident model except those named in ``keep``.

    A dyno point measured while a neighbour still holds VRAM is not a
    measurement of this model — it is a measurement of the leftovers.  The same
    is true in ordinary use: a box whose Ollama is pinned ``keep_alive:
    Forever`` will happily hand the agent a heavily-offloaded flame because
    something else got there first.

    Returns the models actually asked to unload.  Never raises: clearing the
    field is a courtesy, and a server that refuses is the caller's problem to
    notice, not this function's to crash on.
    """
    host = host or default_host()
    keep = set(keep)
    released = []
    for m in loaded_split(host):
        name = m["model"]
        if name in keep:
            continue
        try:
            _http(host + "/api/generate",
                  {"model": name, "keep_alive": 0, "prompt": ""}, timeout=120)
            released.append(name)
        except Exception:
            pass
    return released


def generate_raw(model: str, prompt: str, num_ctx: Optional[int] = None, *,
                 host: Optional[str] = None, num_predict: int = PROBE_TOKENS,
                 keep_alive: Any = "30s", num_gpu: Optional[int] = None) -> dict:
    """One raw generation, returning Ollama's full reply *with its timings*.

    The ordinary chat path hands back only text; the dyno needs ``eval_count``
    and ``eval_duration`` to compute honest tok/s, so it comes here for the
    unabridged answer.

    ``num_ctx=None`` leaves the context unset so the server keeps its own
    default — which is what a caller that merely wants the model *resident*
    should ask for, rather than inventing a window.
    """
    options: dict[str, Any] = {"num_predict": num_predict, "temperature": 0.3}
    if num_ctx:
        options["num_ctx"] = num_ctx
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    return _http((host or default_host()) + "/api/generate",
                 {"model": model, "prompt": prompt, "stream": False,
                  "keep_alive": keep_alive, "options": options})


def read_gpu() -> Optional[dict]:
    """Name and memory of the first CUDA GPU, or None when nvidia-smi is absent.

    Recorded on the curve so a measurement always names the silicon it was
    taken on.  Absence is not an error — plenty of boxes run Ollama on CPU or
    on non-NVIDIA hardware, and they deserve a curve too.
    """
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    if not out:
        return None
    parts = [p.strip() for p in out[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        return {"name": parts[0], "vram_gb": round(float(parts[1]) / 1024, 1)}
    except ValueError:
        return None


# --- judgment: pure reasoning over a measured power curve ----------------


def recommend_num_ctx(curve: list[dict], min_gpu_frac: float = MIN_GPU_FRAC,
                      ceiling: Optional[int] = None) -> Optional[dict]:
    """The largest measured context whose resident fraction holds the band.

    When no point holds the band at all — a model simply too large for this
    card at any useful context — the band has already failed as a guide, and
    residence becomes actively misleading: the ancestor measured a forced split
    wearing 96% "residence" at a fifth of the speed.  Below the band we
    therefore steer by measured truth instead: among points within
    ``SPEED_TOLERANCE`` of the fastest, take the deepest context — speed first,
    and depth to ponder whenever depth is speed-free.

    This is **not** an endorsement.  Callers must check the returned point's
    ``gpu_frac`` against the floor before treating it as one.  Returns a curve
    point, or None for an empty curve.
    """
    if not curve:
        return None
    in_band = [p for p in curve
               if p["gpu_frac"] >= min_gpu_frac
               and (ceiling is None or p["num_ctx"] <= ceiling)]
    if in_band:
        return max(in_band, key=lambda p: (p["num_ctx"], p.get("tok_s", 0)))
    eligible = [p for p in curve if ceiling is None or p["num_ctx"] <= ceiling] or curve
    fastest = max(p.get("tok_s", 0) for p in eligible)
    near = [p for p in eligible if p.get("tok_s", 0) >= fastest * SPEED_TOLERANCE]
    return max(near, key=lambda p: (p["num_ctx"], p.get("tok_s", 0)))


def check_plan(profile: dict, num_ctx: int,
               min_gpu_frac: float = MIN_GPU_FRAC) -> dict:
    """Would this profile at this context hold the band?  Judged from its curve.

    ``status`` is ``"unknown"`` when the model was never put on the dyno — we
    are flying blind, and we say so rather than guess.
    """
    power = (profile or {}).get("power") or {}
    curve = power.get("curve")
    if not curve:
        return {"status": "unknown", "gpu_frac": None, "tok_s": None,
                "recommended_num_ctx": None,
                "note": "no power curve; run the dyno before trusting the band"}
    point = min(curve, key=lambda p: abs(p["num_ctx"] - num_ctx))
    rec = recommend_num_ctx(curve, min_gpu_frac)
    return {"status": "in_band" if point["gpu_frac"] >= min_gpu_frac else "spills",
            "gpu_frac": point["gpu_frac"], "tok_s": point["tok_s"],
            "recommended_num_ctx": rec["num_ctx"] if rec else None,
            "note": f"judged from nearest measured point num_ctx={point['num_ctx']}"}


# --- profiles ------------------------------------------------------------


def operating_num_ctx(profile: Optional[dict]) -> Optional[int]:
    """The context to actually run at.

    The bench *proposes* (``power.recommended_num_ctx``); a human *chooses*
    (``chosen_num_ctx``).  A choice always wins, because the bench measures
    only throughput and a person may reasonably weigh room-to-think against
    it — as we did, taking 64K on a card where 32K measured faster.

    Returns None when the model has never been measured and nobody has
    chosen, which callers must read as "no opinion", never as a default.
    """
    if not profile:
        return None
    chosen = profile.get("chosen_num_ctx")
    if chosen:
        return int(chosen)
    return (profile.get("power") or {}).get("recommended_num_ctx")


def profile_slug(model: str) -> str:
    """Filename-safe name for a model tag (``qwen3.6:latest`` → ``qwen3.6-latest``)."""
    text = "".join(c.lower() if (c.isalnum() or c == ".") else "-" for c in str(model))
    return "-".join(p for p in text.split("-") if p) or "model"


def load_profile(model: str) -> Optional[dict]:
    """This box's measurement of a model, else the one we shipped, else None.

    Precedence is the whole point: our curve describes *our* box.  Someone
    running this fork on different silicon should see their own numbers the
    moment they measure, without deleting anything of ours.
    """
    slug = profile_slug(model)
    for directory in (measured_profile_dir(), SEEDED_PROFILE_DIR):
        path = directory / f"{slug}.json"
        try:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def save_profile(profile: dict, directory: Optional[Path] = None) -> Path:
    """Record a measurement.  Defaults to this user's own dyno directory."""
    directory = directory or measured_profile_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile_slug(profile['model'])}.json"
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return path


# --- the bench -----------------------------------------------------------


def run_dyno(model: str, measure_fn: Callable, *,
             ctx_points: Iterable[int] = DEFAULT_CTX_POINTS,
             min_gpu_frac: float = MIN_GPU_FRAC,
             host: Optional[str] = None, gpu: Optional[dict] = None,
             server_env: Optional[dict] = None, repeats: int = 1,
             warmup: int = 1) -> dict:
    """Sweep one model, build its power curve, and judge it.

    ``measure_fn(model, num_ctx)`` returns ``{"resident_gb", "gpu_frac",
    "tok_s", "prompt_tok_s", "load_s"}``, or None to skip a point the model
    cannot run.

    ``repeats`` rolls each point that many times and records the mean.  A
    single probe wanders run to run by more than the differences the
    recommendation must judge, so a curve meant to decide anything should roll
    at least twice.  Points that error are dropped from the mean; a point that
    never succeeds is skipped entirely rather than recorded as a zero.

    ``warmup`` rolls are taken first and **thrown away**.  Measured on this box
    (2026-07-18, qwen3.6 on a 5070 Ti), the first roll after a cleared field
    ran ~35% slower than the next at every single context — a model freshly
    paged off disk into a partly-offloaded split is not yet the model you will
    actually use.  Averaging that cold roll together with warm ones produces a
    number describing neither, so it is discarded rather than blended.

    Returns the full profile dict (model + power block).  Writing it is the
    caller's choice — see :func:`save_profile`.
    """
    curve = []
    for num_ctx in ctx_points:
        for _ in range(max(0, warmup)):
            measure_fn(model, num_ctx)
        rolls = [m for m in (measure_fn(model, num_ctx) for _ in range(max(1, repeats)))
                 if m is not None]
        if not rolls:
            continue

        def mean(key: str) -> float:
            return sum(r.get(key, 0.0) for r in rolls) / len(rolls)

        point = {"num_ctx": num_ctx,
                 "resident_gb": round(mean("resident_gb"), 2),
                 "gpu_frac": round(mean("gpu_frac"), 3),
                 "tok_s": round(mean("tok_s"), 1),
                 "prompt_tok_s": round(mean("prompt_tok_s"), 1),
                 "load_s": round(mean("load_s"), 1)}
        if len(rolls) > 1:
            # The spread is not decoration. On a partly-offloaded model this
            # box measured rolls 30% apart at one context — wider than the gap
            # between contexts — so a reader who sees only the mean would
            # believe a difference the data cannot support.
            speeds = [r.get("tok_s", 0.0) for r in rolls]
            point["rolls"] = len(rolls)
            point["tok_s_min"] = round(min(speeds), 1)
            point["tok_s_max"] = round(max(speeds), 1)
        curve.append(point)

    rec = recommend_num_ctx(curve, min_gpu_frac)
    return {
        "model": model,
        "power": {
            "dyno_version": DYNO_VERSION,
            "date": time.strftime("%Y-%m-%d"),
            "host": host,
            "gpu": gpu,
            "server_env": server_env,
            "min_gpu_frac": min_gpu_frac,
            "warmup_rolls": max(0, warmup),
            "curve": curve,
            "recommended_num_ctx": rec["num_ctx"] if rec else None,
            "recommended_tok_s": rec["tok_s"] if rec else None,
            "recommended_in_band": bool(rec) and rec["gpu_frac"] >= min_gpu_frac,
            "recommended_within_noise_of": contexts_within_noise(curve, rec),
            "fully_fits": bool(rec) and rec["gpu_frac"] >= 0.999,
        },
    }


def contexts_within_noise(curve: list[dict], rec: Optional[dict]) -> list[int]:
    """Other contexts whose measured speed range overlaps the recommendation's.

    A non-empty list means the bench could not actually separate those
    contexts: the recommendation is one defensible reading of the data rather
    than a finding.  Callers that want a decisive answer should roll more.

    Empty when the curve carries no spread (a single roll per point measures
    nothing about its own reliability, and silence is honester than a
    fabricated confidence).
    """
    if not rec or "tok_s_min" not in rec:
        return []
    low, high = rec["tok_s_min"], rec["tok_s_max"]
    return sorted(p["num_ctx"] for p in curve
                  if p["num_ctx"] != rec["num_ctx"] and "tok_s_min" in p
                  and p["tok_s_min"] <= high and p["tok_s_max"] >= low)


def live_measure(host: Optional[str] = None, settle_s: float = 1.0) -> Callable:
    """The real roller: clear the field, run one timed generation, read the split.

    A generation that errors (context beyond the model's window, server out of
    memory) yields None, so the sweep skips that point instead of recording a
    false zero.
    """
    host = host or default_host()

    def measure(model: str, num_ctx: int) -> Optional[dict]:
        clear_field(host)
        time.sleep(settle_s)  # VRAM is not free the instant the API returns
        try:
            resp = generate_raw(model, PROBE_PROMPT, num_ctx, host=host)
        except Exception:
            return None
        if resp.get("error"):
            return None
        size, vram = model_split(model, host)
        if not size:
            return None
        eval_dur = resp.get("eval_duration", 0) / 1e9
        prompt_dur = resp.get("prompt_eval_duration", 0) / 1e9
        return {
            "resident_gb": size / (1024 ** 3),
            "gpu_frac": gpu_fraction(size, vram),
            "tok_s": (resp.get("eval_count", 0) / eval_dur) if eval_dur else 0.0,
            "prompt_tok_s": ((resp.get("prompt_eval_count", 0) / prompt_dur)
                             if prompt_dur else 0.0),
            "load_s": resp.get("load_duration", 0) / 1e9,
        }

    return measure


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agent.dyno",
        description="Measure a local model's power curve and recommend its context.")
    parser.add_argument("--model", help="model tag to sweep, e.g. qwen3.6:latest")
    parser.add_argument("--host", default=None, help="Ollama host (default: $OLLAMA_HOST)")
    parser.add_argument("--ctx", type=int, nargs="+", default=list(DEFAULT_CTX_POINTS),
                        help="context points to sweep")
    parser.add_argument("--repeats", type=int, default=1,
                        help="counted rolls per point; >=2 for a curve meant to decide anything")
    parser.add_argument("--choose", type=int, default=None, metavar="N",
                        help="run at this context regardless of what the bench "
                             "recommends, and record the choice")
    parser.add_argument("--why", default="", metavar="TEXT",
                        help="the reason for --choose, recorded beside it")
    parser.add_argument("--warmup", type=int, default=1,
                        help="discarded rolls before the counted ones (default 1); "
                             "the first roll after a cleared field runs cold")
    parser.add_argument("--list", action="store_true", help="list installed models and exit")
    parser.add_argument("--seed", action="store_true",
                        help="write into the tracked dyno_profiles/ instead of Hermes home")
    parser.add_argument("--server-env", action="append", default=[], metavar="K=V",
                        help="record a knob the server runs under, e.g. "
                             "OLLAMA_FLASH_ATTENTION=1. Needed when the server is "
                             "started by systemd, whose Environment= lines this "
                             "process cannot see.")
    args = parser.parse_args(argv)

    host = args.host or default_host()
    version = server_version(host)
    if version is None:
        print(f"no Ollama answering at {host}")
        if (os.environ.get("OLLAMA_HOST") or "").strip():
            print(f"  (resolved from OLLAMA_HOST={os.environ['OLLAMA_HOST']!r}; "
                  f"pass --host to override)")
        return 1

    if args.list or not args.model:
        print(f"Ollama {version} at {host}")
        for m in installed_models(host) or []:
            print(f"  {m}")
        for m in loaded_split(host):
            print(f"resident: {m['model']} {m['resident_gb']:.1f} GB "
                  f"gpu={m['gpu_frac']:.0%}")
        return 0

    gpu = read_gpu()
    print(f"Ollama {version} at {host}"
          + (f" | {gpu['name']} {gpu['vram_gb']} GB" if gpu else " | no NVIDIA GPU seen"))
    print(f"sweeping {args.model} across {args.ctx} ({args.repeats} roll(s) each)\n")

    measure = live_measure(host)

    def announce(model: str, num_ctx: int):
        print(f"  num_ctx={num_ctx:>7} ... ", end="", flush=True)
        result = measure(model, num_ctx)
        print("skipped (model refused it)" if result is None
              else f"{result['tok_s']:6.1f} tok/s  gpu={result['gpu_frac']:.0%}  "
                   f"{result['resident_gb']:.1f} GB")
        return result

    env = server_env(host)
    for pair in args.server_env:
        key, _, value = pair.partition("=")
        if key:
            env["env"][key.strip()] = value.strip()

    profile = run_dyno(
        args.model, announce, ctx_points=args.ctx, host=host, gpu=gpu,
        server_env=env, repeats=args.repeats, warmup=args.warmup,
    )
    # A human's choice outlives any single measurement: re-running the bench
    # must not silently revert an operating context somebody settled on.
    existing = load_profile(args.model) or {}
    if args.choose:
        profile["chosen_num_ctx"] = args.choose
        if args.why:
            profile["chosen_because"] = args.why
    elif existing.get("chosen_num_ctx"):
        profile["chosen_num_ctx"] = existing["chosen_num_ctx"]
        if existing.get("chosen_because"):
            profile["chosen_because"] = existing["chosen_because"]

    power = profile["power"]
    path = save_profile(profile, SEEDED_PROFILE_DIR if args.seed else None)

    print(f"\nrecommended num_ctx: {power['recommended_num_ctx']} "
          f"at {power['recommended_tok_s']} tok/s")
    if profile.get("chosen_num_ctx"):
        print(f"chosen num_ctx: {profile['chosen_num_ctx']} (a choice overrides "
              f"the recommendation)")
    if not power["recommended_in_band"]:
        print(f"  NOT in band (<{int(MIN_GPU_FRAC * 100)}% resident) - chosen on "
              f"measured speed, not residence. This model spills on this card.")
    overlaps = power["recommended_within_noise_of"]
    if overlaps:
        print(f"  within measurement noise of {overlaps} - the rolls overlap, so this "
              f"is a reading of the data, not a finding. Roll more to separate them.")
    print(f"written to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
