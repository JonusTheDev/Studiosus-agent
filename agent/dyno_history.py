"""Live throughput history — the real-world companion to the Dyno bench.

The Dyno (``agent/dyno.py``) is a *synthetic* probe: a fixed prompt swept across
context sizes on a quiet machine.  It tells you what a model *can* do.  This
module records what a model *actually did* — the observed generation speed of
real turns — so a model is never trusted on stale or optimistic bench numbers
once it meets real work.

The covenant is **low cost**.  Nothing here calls a model or probes a server.
It appends one line of numbers already computed at turn's end, and reads them
back for the desktop's model-details rail.  A failure never touches the turn:
:func:`record_sample` swallows everything.

Storage sits beside the measured Dyno profiles under ``$HERMES_HOME/dyno`` —
untracked runtime data.  The raw samples may age out; the *summary* is cheap to
recompute and carries no covenant of permanence (unlike a distilled Chronicle
lesson).  One append-only JSONL file per model, capped to the most recent
``MAX_SAMPLES`` so it can never grow without bound.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent import dyno


MAX_SAMPLES = 300      # a rolling window; the tail is trimmed on every write
TREND_POINTS = 20      # how many recent tok/s the summary hands the sparkline
RECENT_WINDOW = 50     # how many recent samples the "recent average" spans


def _log():
    import logging
    return logging.getLogger(__name__)


def _now_ts() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"


def history_dir() -> Path:
    return dyno.measured_profile_dir() / "history"


def _path(model: str) -> Path:
    return history_dir() / f"{dyno.profile_slug(model)}.jsonl"


def record_sample(model: str, *, tok_s: float, num_ctx: Optional[int] = None,
                  source: str = "wallclock",
                  completion_tokens: Optional[int] = None) -> bool:
    """Append one observed-throughput sample.  Never raises.

    ``source`` is ``"native"`` when the figure came from Ollama's own
    ``eval_count``/``eval_duration`` telemetry, or ``"wallclock"`` when it is a
    proxy derived from tokens-over-elapsed.  Returns True on success, False on
    any failure (a sampling failure must never break a turn).
    """
    try:
        if not model or tok_s is None or tok_s <= 0:
            return False
        sample = {
            "ts": _now_ts(),
            "tok_s": round(float(tok_s), 1),
            "num_ctx": int(num_ctx) if num_ctx else None,
            "source": source,
            "completion_tokens": (int(completion_tokens)
                                  if completion_tokens else None),
        }
        path = _path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        _trim(path)
        return True
    except Exception as exc:  # never let a sample break the turn
        _log().debug("dyno_history: record_sample failed for %r: %s", model, exc)
        return False


def _trim(path: Path) -> None:
    """Keep only the most recent ``MAX_SAMPLES`` lines.  Best-effort."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= MAX_SAMPLES:
        return
    kept = lines[-MAX_SAMPLES:]
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def read_samples(model: str) -> list[dict]:
    """Every recorded sample for a model, oldest first.  Corrupt lines skipped."""
    path = _path(model)
    if not path.is_file():
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        return []
    return out


def summary(model: str) -> Optional[dict]:
    """A compact view of a model's observed throughput, or None if never seen.

    ``recent_tok_s_avg`` averages the last ``RECENT_WINDOW`` samples (what the
    model is doing *lately*); ``min``/``max`` span the whole retained window;
    ``trend`` is the last ``TREND_POINTS`` tok/s for a sparkline.  A model with
    no samples reports None — no data is not zero.
    """
    samples = read_samples(model)
    rates = [s.get("tok_s") for s in samples
             if isinstance(s.get("tok_s"), (int, float)) and s.get("tok_s") > 0]
    if not rates:
        return None
    recent = rates[-RECENT_WINDOW:]
    return {
        "samples": len(rates),
        "recent_tok_s_avg": round(sum(recent) / len(recent), 1),
        "min": round(min(rates), 1),
        "max": round(max(rates), 1),
        "last_seen": samples[-1].get("ts"),
        "last_source": samples[-1].get("source"),
        "trend": [round(r, 1) for r in rates[-TREND_POINTS:]],
    }
