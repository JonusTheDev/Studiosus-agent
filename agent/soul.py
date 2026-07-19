"""The Soul — an append-only, sealed-with-outcome episodic ledger.

Phase A of the Studiosus graft (``.plans/studiosus-heart-strategy.md``): the
ground truth every later phase reads from.  One JSONL file per episode under
``$HERMES_HOME/soul/``, recording *I thought X, called tool Y, got Z,
delivered W, sealed with outcome O*.

Two covenants carried over from the ancestor Loom, distilled rather than
ported:

- **The past is not edited, not even by its owner.**  Every event line carries
  a monotonically increasing ``seq`` so ordering never depends on timestamp
  resolution, and a sealed episode refuses all further writes.
- **Fumbles are recorded exactly as they happened.**  A failed or abandoned
  turn seals with that outcome; the Soul is honest or it is nothing.

Storage: raw episodes are voluminous and live under the (untracked) Hermes
home, because their *meaning* will be preserved by the tracked Chronicle in
Phase B.  The distilled must never be lost; the raw may age out.

Purely observational, and **off by default** — set ``HERMES_SOUL=1`` to
enable.  Nothing in Hermes reads these files yet, so with the flag off (or on)
behavior is unchanged.  A write failure must never break a turn: the single
call site in ``agent/turn_finalizer.py`` is fully guarded, and so is
:func:`record_turn`.

An episode is one **turn** (keyed by ``turn_id``, which is unique by
construction), and it carries its ``task_id`` and ``session_id`` as fields.
Grouping episodes into larger units of work is therefore a read-time question
for later phases, and never an append to an already-sealed record.

The module is a genuine append-only writer: :meth:`Episode.record` may be
called throughout a turn.  Phase A's only caller happens to write one episode
in a single pass at finalization, which is the lowest-risk hook that still
sees the whole turn — instrumenting the hot loop is a later phase's work.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from utils import is_truthy_value

# The canonical event vocabulary.  Free-form types are allowed (the Soul
# records, it does not police), but these are what the Loom emits.
EPISODE_BEGIN = "episode_begin"
ASSEMBLE = "assemble"
THOUGHT = "thought"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
DELIVERY = "delivery"
EPISODE_END = "episode_end"

# Outcomes an episode may seal with.  ``complete`` is the only one that counts
# toward a skill family in Phase D, so the mapping in :func:`turn_outcome` is
# deliberately strict.
OUTCOME_COMPLETE = "complete"
OUTCOME_FAILED = "failed"
OUTCOME_ABANDONED = "abandoned"
OUTCOME_INCOMPLETE = "incomplete"

MAX_FIELD_CHARS = 2000  # a ledger line is a record, not an archive of the payload


def soul_enabled() -> bool:
    """True when the Soul should record.  Off unless ``HERMES_SOUL`` is truthy."""
    return is_truthy_value(os.environ.get("HERMES_SOUL"))


def soul_dir() -> Path:
    """The episode directory.  Resolved lazily so ``HERMES_HOME`` overrides
    (and the per-context override used by tests) are always honored."""
    return get_hermes_home() / "soul"


def now_ts() -> str:
    """A sortable UTC stamp, millisecond resolution."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"


def _slug(value: Any, limit: int = 80) -> str:
    """Filename-safe slug.  ``turn_id`` is colon-delimited, which is illegal in
    Windows filenames, so ids are always slugged before they touch the disk."""
    text = "".join(c.lower() if c.isalnum() else "-" for c in str(value))
    return "-".join(p for p in text.split("-") if p)[:limit] or "episode"


def _redact(text: str) -> str:
    """Strip credentials from a string bound for the ledger.

    Redaction is forced — these lines land on disk, so they must never carry a
    raw credential regardless of the user's global logging preference.
    """
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        return text  # a redaction failure must not cost us the record


def _sanitize(value: Any) -> Any:
    """Redact every string reachable from an event field.

    Applied inside :meth:`Episode.record`, which is the single chokepoint
    before the disk.  Doing it here rather than at the call sites means a
    caller cannot leak a secret into the Soul by forgetting to clip first —
    including the live-instrumentation callers that later phases will add.
    """
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _clip(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    """Render a field for the ledger with its length bounded.

    Redaction happens in :func:`_sanitize` at write time; this only decides how
    much of the payload the record keeps.
    """
    text = value if isinstance(value, str) else str(value)
    if len(text) > limit:
        return text[:limit] + f"...[clipped {len(text) - limit} chars]"
    return text


class SealedEpisodeError(RuntimeError):
    """Raised on any attempt to write to a sealed episode."""


class Episode:
    """A single turn's living record.  Append-only; sealed when the work ends."""

    def __init__(self, episode_id: str, path: Path):
        self.episode_id = episode_id
        self.path = path
        self.seq = 0
        self.sealed = False

    def record(self, event_type: str, **fields: Any) -> dict:
        """Append one event.  Returns the event dict as written."""
        if self.sealed:
            raise SealedEpisodeError(
                f"episode {self.episode_id} is sealed; the past is not edited"
            )
        self.seq += 1
        event = {"seq": self.seq, "ts": now_ts(), "type": event_type,
                 **{k: _sanitize(v) for k, v in fields.items()}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event

    def seal(self, outcome: str, summary: str = "") -> None:
        """Close the episode with its outcome.

        The final event is written first, then the episode refuses all further
        writes.
        """
        self.record(EPISODE_END, outcome=outcome, summary=_clip(summary, 500))
        self.sealed = True


def begin_episode(
    turn_id: str,
    *,
    task_id: str = "",
    session_id: str = "",
    task_text: str = "",
    **fields: Any,
) -> Episode:
    """Open an episode for a turn and write its first event.

    ``turn_id`` is unique by construction, but the id is still checked against
    the directory: two episodes must never share one file, because a shared
    file would silently weave two memories into one.
    """
    directory = soul_dir()
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{now_ts()}-{_slug(turn_id)}"
    episode_id, n = base, 1
    while (directory / f"{episode_id}.jsonl").exists():
        n += 1
        episode_id = f"{base}-{n}"
    episode = Episode(episode_id, directory / f"{episode_id}.jsonl")
    episode.record(
        EPISODE_BEGIN,
        turn_id=turn_id,
        task_id=task_id,
        session_id=session_id,
        task_text=_clip(task_text),
        **fields,
    )
    return episode


def read_episode(episode_id: str) -> list[dict]:
    """An episode's events in ``seq`` order.  Raises ``FileNotFoundError``."""
    path = soul_dir() / f"{episode_id}.jsonl"
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return sorted(events, key=lambda e: e.get("seq", 0))


def list_episodes() -> list[str]:
    """Episode ids present in the Soul, oldest first (ids embed their stamp)."""
    directory = soul_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.jsonl"))


def is_sealed(episode_id: str) -> bool:
    """True if the stored episode ends with an ``episode_end`` event."""
    events = read_episode(episode_id)
    return bool(events) and events[-1]["type"] == EPISODE_END


def turn_outcome(*, completed: bool, failed: bool, interrupted: bool) -> str:
    """Map Hermes' turn flags onto the Soul's outcome vocabulary.

    Interruption is checked first: a turn the user stopped is *abandoned*, not
    failed — the distinction matters, because Phase D counts only episodes that
    truly sealed ``complete`` and a steward reading the ledger deserves to know
    whether a turn broke or was simply called off.
    """
    if interrupted:
        return OUTCOME_ABANDONED
    if failed:
        return OUTCOME_FAILED
    if completed:
        return OUTCOME_COMPLETE
    return OUTCOME_INCOMPLETE


def _tool_call_name(call: Any) -> str:
    """Tool calls arrive as SDK objects or plain dicts depending on provider."""
    function = getattr(call, "function", None)
    if function is not None:
        return str(getattr(function, "name", "") or "")
    if isinstance(call, dict):
        return str((call.get("function") or {}).get("name", "") or "")
    return ""


def _tool_call_args(call: Any) -> str:
    function = getattr(call, "function", None)
    if function is not None:
        return str(getattr(function, "arguments", "") or "")
    if isinstance(call, dict):
        return str((call.get("function") or {}).get("arguments", "") or "")
    return ""


def turn_events(messages: Iterable[dict]) -> list[tuple[str, dict]]:
    """Derive this turn's events from the working message list.

    Only the current turn is walked: we scan back to the user message that
    started it, exactly as ``finalize_turn`` does when extracting reasoning, so
    nothing from a prior turn leaks into this episode.
    """
    messages = list(messages)
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            start = i + 1
            break

    events: list[tuple[str, dict]] = []
    for msg in messages[start:]:
        role = msg.get("role")
        if role == "assistant":
            if msg.get("reasoning"):
                events.append((THOUGHT, {"text": _clip(msg["reasoning"])}))
            for call in msg.get("tool_calls") or []:
                events.append((TOOL_CALL, {
                    "tool": _tool_call_name(call),
                    "arguments": _clip(_tool_call_args(call)),
                }))
        elif role == "tool":
            events.append((TOOL_RESULT, {
                "tool": str(msg.get("name", "") or ""),
                "result": _clip(msg.get("content", "")),
            }))
    return events


def record_turn(
    *,
    turn_id: str,
    task_id: str = "",
    session_id: str = "",
    user_message: Any = "",
    messages: Optional[Iterable[dict]] = None,
    final_response: Optional[str] = None,
    completed: bool = False,
    failed: bool = False,
    interrupted: bool = False,
    exit_reason: Any = "",
    model: str = "",
    platform: str = "",
    served_lessons: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Write one sealed episode for a finished turn.  Returns its id, or None.

    Returns None (recording nothing) when the flag is off.  Never raises: the
    Soul observes a turn, and an observer that can break the thing it watches
    is worse than no observer at all.
    """
    if not soul_enabled():
        return None
    started = time.monotonic()
    try:
        episode = begin_episode(
            turn_id,
            task_id=task_id,
            session_id=session_id,
            task_text=user_message if isinstance(user_message, str) else "",
        )
        # What was laid beside the work, recorded beside the outcome it led to.
        # This pairing is the whole basis of the Phase D correlation report, and
        # it cannot be reconstructed after the fact — hence from day one.
        episode.record(ASSEMBLE, model=model, platform=platform,
                       lessons=list(served_lessons or []))
        for event_type, fields in turn_events(messages or []):
            episode.record(event_type, **fields)
        if final_response:
            episode.record(DELIVERY, text=_clip(final_response))
        episode.seal(
            turn_outcome(completed=completed, failed=failed, interrupted=interrupted),
            summary=str(exit_reason or ""),
        )
        return episode.episode_id
    except Exception:
        try:
            from agent.conversation_loop import logger
            logger.warning(
                "soul: failed to record episode for turn %s after %.3fs",
                turn_id, time.monotonic() - started, exc_info=True,
            )
        except Exception:
            pass
        return None
