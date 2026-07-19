"""Voice — versioned hats for one head.  Phase E of the Studiosus graft.

Hermes' identity today is one string: ``SOUL.md`` if present, else the
hardcoded default.  The ancestor Loom kept its persona instead as versioned
markdown files — ``<hat>.v<N>.md`` under a tracked ``voices/`` directory —
different hats (student, builder, reviewer) on one head, not different
agents.  Old versions stay on disk rather than being overwritten, because a
revised voice is measured against the prior voice on real work before it is
adopted; a voice is a stance, not a manual, and it honors a strict size
budget.

This module rebuilds that discipline on Hermes' bricks.  ``HERMES_VOICE``
names a hat (``student``) or pins a version (``student.v2``); when it is
unset — the default — Hermes' identity resolution is exactly what it was,
so the graft is inert until asked for.  The chosen voice becomes the
*identity* part of the stable prompt tier, resolved once at prompt build:
static for the whole session, so the upstream prefix cache stays warm
(``references/system-prompt-invariant.md``).

A missing hat never breaks the prompt: selection failures log a warning and
fall back to today's behavior.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Tracked beside the repo, like dyno_profiles/: a voice is given knowledge,
# grown deliberately, and versioned in the open.
VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

_FILE_RE = re.compile(r"^(?P<hat>[a-z0-9_-]+)\.v(?P<version>\d+)\.md$")
_SELECT_RE = re.compile(r"^(?P<hat>[a-z0-9_-]+?)(?:\.v(?P<version>\d+))?$")

# A voice is a stance, not a manual (the ancestor's size-budget lesson).
MAX_VOICE_CHARS = 8000


def list_voices(directory: Optional[Path] = None) -> dict[str, list[int]]:
    """Map of hat -> sorted available versions, e.g. ``{"student": [1, 2]}``."""
    directory = directory or VOICES_DIR
    out: dict[str, list[int]] = {}
    if not directory.is_dir():
        return out
    for p in directory.iterdir():
        m = _FILE_RE.match(p.name)
        if m:
            out.setdefault(m.group("hat"), []).append(int(m.group("version")))
    return {hat: sorted(vs) for hat, vs in sorted(out.items())}


def load_voice(hat: str, version: Optional[int] = None,
               directory: Optional[Path] = None) -> dict:
    """Load a voice by hat name; highest version unless ``version`` pins one.

    Returns ``{"hat", "version", "text", "path"}``.  Raises
    ``FileNotFoundError`` if the hat (or the pinned version) does not exist —
    old versions staying on disk is what makes pinning meaningful.
    """
    directory = directory or VOICES_DIR
    versions = list_voices(directory).get(hat, [])
    if not versions:
        raise FileNotFoundError(
            f"no voice files for hat {hat!r} in {directory}")
    if version is None:
        version = versions[-1]
    elif version not in versions:
        raise FileNotFoundError(
            f"voice {hat}.v{version} not found (have versions {versions})")
    path = directory / f"{hat}.v{version}.md"
    text = path.read_text(encoding="utf-8")
    if len(text) > MAX_VOICE_CHARS:
        # Served anyway — the budget is a covenant for authors, and a warning
        # is the honest enforcement for an identity that must never vanish
        # mid-session because it grew a paragraph too long.
        logger.warning("voice %s.v%d runs %d chars (budget %d): a voice is "
                       "a stance, not a manual", hat, version, len(text),
                       MAX_VOICE_CHARS)
    return {"hat": hat, "version": version, "text": text, "path": str(path)}


def selected_voice(directory: Optional[Path] = None) -> Optional[dict]:
    """The voice ``HERMES_VOICE`` asks for, or None (wear no hat, change
    nothing).

    Accepts ``student`` (highest version) or ``student.v2`` (pinned).  A
    malformed or missing selection logs a warning and returns None so the
    prompt build falls back to Hermes' stock identity — a bad env var must
    never cost a session its face.
    """
    raw = (os.environ.get("HERMES_VOICE") or "").strip().lower()
    if not raw:
        return None
    m = _SELECT_RE.match(raw)
    if not m:
        logger.warning("HERMES_VOICE=%r is not a hat name (want e.g. "
                       "'student' or 'student.v2'); wearing no hat", raw)
        return None
    version = int(m.group("version")) if m.group("version") else None
    try:
        return load_voice(m.group("hat"), version, directory)
    except (FileNotFoundError, OSError) as exc:
        logger.warning("HERMES_VOICE=%r: %s; wearing no hat", raw, exc)
        return None
