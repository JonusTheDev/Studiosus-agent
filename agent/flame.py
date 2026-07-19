"""Tending the flame — make the local model ready, or say plainly why not.

A local setup fails in a handful of dull, recoverable ways: the server is up
but the model was never loaded; something else got there first and holds the
VRAM; the model is resident but at whatever context the last caller happened to
ask for.  None of these are interesting, and all of them produce the same
symptom — the agent is inexplicably slow, or refuses to start.

This module looks at the box, says exactly which of those is true, and (when
asked) fixes the ones that are safe to fix.

**What it will do:** release resident models that are competing for the card,
and load the intended model at the intended context, pinned.

**What it will never do:** pull a model that is not installed (a 24 GB download
is not a self-heal), or start a server that is not running (the flame is tended
here, not owned here).  Both are reported for a human to act on.  Healing that
reaches past what the caller expected is not healing.

The context it aims for comes from the dyno profile — a human's
``chosen_num_ctx`` when one exists, else the bench's recommendation (see
:func:`agent.dyno.operating_num_ctx`).  With no profile at all this module has
no opinion and says so, rather than inventing a number.

Off by default: set ``HERMES_FLAME=1`` to let startup tend the flame.
Releasing another process's models is a real side effect and nobody should get
it by surprise.

    python -m agent.flame --status
    python -m agent.flame --ensure --clear-conflicts
"""

from __future__ import annotations

import os
from typing import Optional

from agent import dyno
from utils import is_truthy_value

# How long a tended flame stays resident.  -1 is Ollama's "forever": a model
# that unloads between turns pays its whole load time again on the next one,
# which on a partly-offloaded 24 GB model is tens of seconds of silence.
PIN_FOREVER = -1
DEFAULT_KEEP_ALIVE = "30m"


def flame_enabled() -> bool:
    """True when startup should tend the flame.  Off unless ``HERMES_FLAME``."""
    return is_truthy_value(os.environ.get("HERMES_FLAME"))


def host_from_base_url(base_url: Optional[str]) -> Optional[str]:
    """Turn an OpenAI-compatible base URL into the Ollama root beneath it.

    Hermes talks to Ollama through ``http://host:11434/v1``; the native
    endpoints this module needs (``/api/ps``, ``/api/generate``) hang off the
    root instead.  Returns None for an empty URL so the caller falls back to
    :func:`agent.dyno.default_host` rather than dialing an empty string.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return None
    for suffix in ("/v1", "/api"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def intended_num_ctx(model: str) -> Optional[int]:
    """The context this model should run at, or None if nobody has an opinion."""
    return dyno.operating_num_ctx(dyno.load_profile(model))


def inspect(model: str, *, num_ctx: Optional[int] = None,
            host: Optional[str] = None) -> dict:
    """Look at the box and report what stands between here and a ready flame.

    Pure observation — changes nothing.  ``ready`` is True only when the model
    is resident at the wanted context with nothing else competing for the card.

    ``installed`` is None (not False) when the server could not be asked: "I
    could not check" and "it is not there" are different facts, and conflating
    them turns a network blip into a missing-model error.
    """
    host = host or dyno.default_host()
    want = num_ctx if num_ctx is not None else intended_num_ctx(model)

    report = {
        "host": host, "model": model, "want_num_ctx": want,
        "reachable": False, "installed": None, "loaded": False,
        "loaded_num_ctx": None, "pinned": False, "gpu_frac": None,
        "conflicts": [], "ready": False, "blockers": [],
    }

    version = dyno.server_version(host)
    if version is None:
        report["blockers"].append(f"no Ollama answering at {host}")
        return report
    report["reachable"] = True
    report["server_version"] = version

    installed = dyno.installed_models(host)
    if installed is not None:
        report["installed"] = model in installed
        if not report["installed"]:
            report["blockers"].append(
                f"{model} is not installed on this server (pull it first; "
                f"this is not something to download behind your back)")

    for resident in _resident(host):
        if resident["model"] == model:
            report["loaded"] = True
            report["loaded_num_ctx"] = resident.get("context_length")
            report["gpu_frac"] = resident.get("gpu_frac")
            report["pinned"] = resident.get("pinned", False)
        else:
            report["conflicts"].append(resident["model"])

    if report["conflicts"]:
        report["blockers"].append(
            "competing for the card: " + ", ".join(report["conflicts"]))
    if not report["loaded"]:
        report["blockers"].append(f"{model} is not resident")
    elif want and report["loaded_num_ctx"] and report["loaded_num_ctx"] != want:
        report["blockers"].append(
            f"resident at num_ctx={report['loaded_num_ctx']}, wanted {want}")

    report["ready"] = report["loaded"] and not report["conflicts"] and (
        not want or not report["loaded_num_ctx"]
        or report["loaded_num_ctx"] == want)
    return report


def _resident(host: str) -> list[dict]:
    """Resident models with the fields ``/api/ps`` carries beyond the split."""
    try:
        models = dyno._http(host.rstrip("/") + "/api/ps", timeout=30).get("models", [])
    except Exception:
        return []
    out = []
    for m in models:
        size = m.get("size", 0)
        expires = str(m.get("expires_at") or "")
        out.append({
            "model": m.get("name") or m.get("model"),
            "context_length": m.get("context_length"),
            "gpu_frac": dyno.gpu_fraction(size, m.get("size_vram", 0)),
            # Ollama writes a far-future expiry for keep_alive=-1 rather than a
            # flag, so "pinned" is read from the year rather than asked for.
            "pinned": expires[:4].isdigit() and int(expires[:4]) > 2100,
        })
    return out


def ensure_ready(model: str, *, num_ctx: Optional[int] = None,
                 host: Optional[str] = None, clear_conflicts: bool = False,
                 pin: bool = True) -> dict:
    """Bring the flame to ready if it can be done safely, and report what was done.

    Returns the post-action :func:`inspect` report with an ``actions`` list.
    Never raises: a startup helper that can abort startup is worse than one
    that reports a cold flame and lets the turn proceed.
    """
    host = host or dyno.default_host()
    report = inspect(model, num_ctx=num_ctx, host=host)
    actions: list[str] = []

    # Nothing below can help an unreachable server or an absent model, and
    # neither is ours to fix without asking.
    if not report["reachable"] or report["installed"] is False:
        report["actions"] = actions
        return report

    try:
        if report["conflicts"] and clear_conflicts:
            released = dyno.clear_field(host, keep={model})
            if released:
                actions.append("released " + ", ".join(released))

        want = report["want_num_ctx"]
        needs_load = not report["loaded"] or (
            want and report["loaded_num_ctx"] and report["loaded_num_ctx"] != want)
        if needs_load:
            # An empty prompt with num_predict=0 makes Ollama load the model
            # and generate nothing — residence is the whole point here.
            dyno.generate_raw(
                model, "", want, host=host, num_predict=0,
                keep_alive=PIN_FOREVER if pin else DEFAULT_KEEP_ALIVE,
            )
            actions.append(
                f"loaded {model}" + (f" at num_ctx={want}" if want else "")
                + (" (pinned)" if pin else ""))
    except Exception as exc:
        actions.append(f"failed while tending: {exc}")

    final = inspect(model, num_ctx=num_ctx, host=host)
    final["actions"] = actions
    return final


def describe(report: dict) -> str:
    """One human-readable line (plus reasons) for a status report."""
    if report["ready"]:
        ctx = report.get("loaded_num_ctx")
        gpu = report.get("gpu_frac")
        return (f"flame ready: {report['model']}"
                + (f" at num_ctx={ctx}" if ctx else "")
                + (f", {gpu:.0%} on GPU" if gpu is not None else "")
                + (" (pinned)" if report.get("pinned") else ""))
    return (f"flame not ready: {report['model']} - "
            + "; ".join(report["blockers"] or ["unknown"]))


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agent.flame",
        description="Check, and optionally restore, the local model's readiness.")
    parser.add_argument("--model", required=True, help="model tag, e.g. qwen3.6:latest")
    parser.add_argument("--host", default=None, help="Ollama host (default: $OLLAMA_HOST)")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="context to aim for (default: the dyno profile's)")
    parser.add_argument("--ensure", action="store_true",
                        help="fix what is safely fixable, don't just report")
    parser.add_argument("--clear-conflicts", action="store_true",
                        help="release other resident models competing for the card")
    parser.add_argument("--no-pin", action="store_true",
                        help="let the model expire normally instead of pinning it")
    args = parser.parse_args(argv)

    if args.ensure:
        report = ensure_ready(args.model, num_ctx=args.num_ctx, host=args.host,
                              clear_conflicts=args.clear_conflicts,
                              pin=not args.no_pin)
        for action in report.get("actions", []):
            print(f"  {action}")
    else:
        report = inspect(args.model, num_ctx=args.num_ctx, host=args.host)

    print(describe(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
