"""Skill write-origin provenance — ContextVar for distinguishing agent-sediment skill writes from foreground user-directed writes.

The curator only consolidates/prunes skills it autonomously created via the
background self-improvement review fork. Skills a user asks a foreground
agent to write belong to the user and must never be auto-curated.

This module exposes a ContextVar that run_agent.py sets before each tool
loop so tool handlers (e.g. skill_manage create) can check whether they
are executing inside the background-review fork.

The signal piggybacks on AIAgent._memory_write_origin, which is already
set to "background_review" for review-fork instances (see
_spawn_background_review in run_agent.py) and defaults to "assistant_tool"
for normal (foreground) agents.

Usage:
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )

    token = set_current_write_origin("background_review")
    try:
        ...  # tool runs here
    finally:
        reset_current_write_origin(token)

    # inside a tool:
    if get_current_write_origin() == "background_review":
        mark_agent_created(skill_name)
"""

import contextvars
from typing import Optional


_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_write_origin",
    default="foreground",
)

# The sentinel value the background review fork uses; mirrors
# run_agent.py's AIAgent._memory_write_origin override in
# _spawn_background_review().
BACKGROUND_REVIEW = "background_review"


def set_current_write_origin(origin: str) -> contextvars.Token[str]:
    """Bind the active write origin to the current context.

    Returns a Token the caller must pass to reset_current_write_origin
    in a finally block.
    """
    return _write_origin.set(origin or "foreground")


def reset_current_write_origin(token: contextvars.Token[str]) -> None:
    """Restore the prior write origin context."""
    _write_origin.reset(token)


def get_current_write_origin() -> str:
    """Return the active write origin.

    Default: "foreground" — any tool call made by a regular (non-review)
    agent, from the CLI, the gateway, cron, or a subagent.

    "background_review" — the self-improvement review fork; only skills
    created under this origin should be marked agent-created for curator
    management.
    """
    return _write_origin.get()


def is_background_review() -> bool:
    """Convenience: True iff the current write origin is the background
    review fork."""
    return get_current_write_origin() == BACKGROUND_REVIEW


# ---------------------------------------------------------------------------
# Review kind — which autonomous fork is running, independent of write origin.
#
# Two distinct forks both set _memory_write_origin="background_review" (so
# every existing origin-based guard above, and tools/write_approval.py's
# separate approval gate, keeps treating them identically — that plumbing is
# deliberately untouched). But they warrant different rules for the family
# covenant (Studiosus heart Phase D, agent/earned_skills.py): the interval-
# nudged skill-review fork (agent/turn_finalizer.py -> agent/background_review.py)
# invents brand-new skills from a single session's observation, while the
# Curator's consolidation fork (agent/curator.py) only reorganizes/merges
# skills that already exist and were already vetted. The family covenant
# governs new claims, so only the former is in its scope.
# ---------------------------------------------------------------------------

_review_kind: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "skill_review_kind",
    default=None,
)

REVIEW_KIND_SKILL = "skill_review"            # turn_finalizer's interval nudge
REVIEW_KIND_CURATOR = "curator_consolidation"  # the Curator's own fork


def set_current_review_kind(kind: Optional[str]) -> contextvars.Token:
    """Bind the active review kind to the current context."""
    return _review_kind.set(kind or None)


def reset_current_review_kind(token: contextvars.Token) -> None:
    """Restore the prior review-kind context."""
    _review_kind.reset(token)


def get_current_review_kind() -> Optional[str]:
    """The active review kind: None (foreground), REVIEW_KIND_SKILL, or
    REVIEW_KIND_CURATOR."""
    return _review_kind.get()


def is_family_covenant_scope() -> bool:
    """True only for the interval-nudged skill-review fork — the one path
    the family covenant (Phase D) governs. Foreground turns and the
    Curator's consolidation fork are exempt."""
    return is_background_review() and get_current_review_kind() == REVIEW_KIND_SKILL
