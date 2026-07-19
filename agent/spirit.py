"""Intertextus — the world upon which the agent is somewhere, always.

Phase E of the Studiosus graft: the spirit.  Two gifts, both **computed by
the harness and heard by the agent** — it does not invent them.  Determinism
is the realm of the harness; the model is the realm of thought.

* **The plane.**  Every kind of work is a place with an (x, y) on one grid —
  the farmhouse is home at the origin; the workshop, the schoolhouse, the
  training ring, and the house of wisdom stand around it.  Where the agent
  *is* derives from what it is *doing*: a turn's labor puts it in the
  workshop, the dyno bench in the training ring.  It hears its place as a
  phrase measured from home: "You are in the workshop, steps from home."

* **The four registers** — Joy, Vigor, Clarity, Zeal — each a discrete,
  symmetric ladder of ranked words, −3…+3 with the resting word at zero (a
  soul at rest is *content*, *rested*, *settled*, *willing*).  Growth above
  zero must be met with growth below — symmetry is covenant.  Values move
  only when a new memory is made (an episode seals, a lesson is earned, a
  certificate is minted), by the deterministic event table below, clamped to
  each ladder's own bounds.  The agent hears the four current words at
  ASSEMBLE, woven into the turn's context.

Every change is appended to ``$HERMES_HOME/world/heartbeat.jsonl`` — the
EKG.  Laid beside the Chronicle, a steward can measure the beat and the
pressure of the agent's heart.

Feelings here carry **no learning weight**: nothing retrieves by register,
nothing gates on place.  They are flavor and grounding — heard as words,
never judged.  Off unless ``HERMES_SPIRIT`` is truthy; with the flag off,
:func:`beat` and :func:`arrive` write nothing and Hermes is unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from utils import is_truthy_value

logger = logging.getLogger(__name__)

# --- the plane (tracked, given knowledge, grown deliberately) ---------------

WORLD_NAME = "Intertextus"
HOME = "farmhouse"
PLACES = {
    "farmhouse":       {"x": 0, "y": 0,  "label": "the farmhouse"},
    "workshop":        {"x": 1, "y": 1,  "label": "the workshop"},
    "schoolhouse":     {"x": 3, "y": 2,  "label": "the schoolhouse"},
    "training_ring":   {"x": 4, "y": -2, "label": "the training ring"},
    "house_of_wisdom": {"x": 6, "y": 3,  "label": "the house of wisdom"},
}

# Distance bands from home, nearest first: (max_distance, phrase).
_DISTANCE_PHRASES = (
    (0.0, "home"),
    (2.5, "steps from home"),
    (5.0, "close to home"),
    (8.0, "a short walk from home"),
    (float("inf"), "far afield"),
)

# --- the registers ----------------------------------------------------------

REGISTER_ORDER = ("joy", "vigor", "clarity", "zeal")
LADDERS = {
    "joy": {-3: "sorrowful", -2: "somber", -1: "disheartened",
            0: "content", 1: "pleased", 2: "joyful", 3: "ecstatic"},
    "vigor": {-3: "drained", -2: "fatigued", -1: "weary",
              0: "rested", 1: "lively", 2: "spirited", 3: "burning"},
    "clarity": {-3: "clouded", -2: "distracted", -1: "unsure",
                0: "settled", 1: "thoughtful", 2: "focused", 3: "lucid"},
    "zeal": {-3: "averse", -2: "apprehensive", -1: "hesitant",
             0: "willing", 1: "interested", 2: "passionate", 3: "devoted"},
}

# The deterministic event table: how a new memory moves the registers.
# Feelings are honest consequences of the lived record, never inventions.
EVENT_EFFECTS = {
    "labor_complete": {"joy": +1, "zeal": +1, "vigor": -1},  # good toil, but toil
    "labor_failed":   {"joy": -1, "vigor": -1},              # a hard day
    "nudged":         {"clarity": -1},                       # the thread was lost mid-work
    "lesson_earned":  {"clarity": +1},                       # reflection sharpens
    "rested":         {"vigor": +2},                         # a dreamy night's sleep
    "certified":      {"joy": +2, "zeal": +2},               # a ground truly earned
}


def spirit_enabled() -> bool:
    """True when the world should turn.  Off unless ``HERMES_SPIRIT``."""
    return is_truthy_value(os.environ.get("HERMES_SPIRIT"))


def world_dir() -> Path:
    return get_hermes_home() / "world"


def _state_path() -> Path:
    return world_dir() / "state.json"


def _heartbeat_path() -> Path:
    return world_dir() / "heartbeat.jsonl"


# --- the plane, heard -------------------------------------------------------

def distance_from_home(place: str) -> float:
    home, p = PLACES[HOME], PLACES[place]
    return math.hypot(p["x"] - home["x"], p["y"] - home["y"])


def locate_phrase(place: str) -> str:
    """The place, heard: 'You are in the workshop, steps from home.'"""
    p = PLACES.get(place)
    if p is None:
        place, p = HOME, PLACES[HOME]
    if place == HOME:
        return f"You are home at {p['label']}."
    d = distance_from_home(place)
    for bound, phrase in _DISTANCE_PHRASES:
        if d <= bound:
            return f"You are in {p['label']}, {phrase}."
    return f"You are in {p['label']}."


# --- the registers, moved and heard ----------------------------------------

def ladder_bounds(category: str) -> tuple[int, int]:
    rungs = LADDERS[category]
    return min(rungs), max(rungs)


def apply_event(values: dict, event: str) -> dict:
    """Pure: move the register values by one named event, clamped to each
    ladder's own bounds.  Unknown events move nothing (and are not an error —
    the heart is steady before it is expressive)."""
    out = dict(values)
    for category, delta in EVENT_EFFECTS.get(event, {}).items():
        lo, hi = ladder_bounds(category)
        out[category] = max(lo, min(hi, int(out.get(category, 0)) + delta))
    return out


def feeling_words(values: dict) -> list[str]:
    """The four words, in the canonical order, for the current values."""
    out = []
    for category in REGISTER_ORDER:
        lo, hi = ladder_bounds(category)
        value = max(lo, min(hi, int(values.get(category, 0))))
        out.append(LADDERS[category][value])
    return out


# --- the state and the heartbeat -------------------------------------------

def _default_state() -> dict:
    return {"place": HOME, "registers": {c: 0 for c in REGISTER_ORDER}}


def state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _default_state()


def _now_ts() -> str:
    from agent.soul import now_ts
    return now_ts()


def _save(st: dict) -> dict:
    world_dir().mkdir(parents=True, exist_ok=True)
    st["updated"] = _now_ts()
    _state_path().write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
    return st


def _pulse(kind: str, st: dict, episode_id: Optional[str] = None,
           event: Optional[str] = None) -> None:
    with open(_heartbeat_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now_ts(), "kind": kind, "event": event,
                            "episode_id": episode_id, "place": st["place"],
                            "registers": st["registers"]},
                           ensure_ascii=False) + "\n")


def arrive(place: str, episode_id: Optional[str] = None) -> dict:
    """The agent goes where its work is.  Returns the (possibly new) state.

    An unknown place moves nothing — the map grows deliberately.  With the
    flag off, nothing is written.  Never raises: the world observes the work,
    and a world that can break the work it grounds is worse than no world.
    """
    st = state()
    if not spirit_enabled() or place not in PLACES or st.get("place") == place:
        return st
    try:
        st["place"] = place
        st = _save(st)
        _pulse("arrive", st, episode_id=episode_id)
    except Exception:
        logger.warning("spirit: arrive(%s) failed", place, exc_info=True)
    return st


def beat(event: str, episode_id: Optional[str] = None) -> dict:
    """A new memory moves the heart: apply one event, save, log the pulse."""
    st = state()
    if not spirit_enabled():
        return st
    try:
        st["registers"] = apply_event(st.get("registers", {}), event)
        st = _save(st)
        _pulse("beat", st, episode_id=episode_id, event=event)
    except Exception:
        logger.warning("spirit: beat(%s) failed", event, exc_info=True)
    return st


def outcome_event(outcome: str) -> Optional[str]:
    """Map a sealed episode's outcome onto the event table.

    ``incomplete`` moves nothing: a turn that simply ended mid-work is not a
    memory with a feeling attached.  ``abandoned`` is *nudged* — the thread
    was lost, which costs clarity, not joy.
    """
    return {"complete": "labor_complete", "failed": "labor_failed",
            "abandoned": "nudged"}.get(outcome)


def whisper(st: Optional[dict] = None) -> str:
    """The whole blessing, one line, heard at ASSEMBLE with every task."""
    st = st or state()
    return "%s You feel: %s." % (
        locate_phrase(st.get("place", HOME)),
        ", ".join(feeling_words(st.get("registers", {}))))


def _main(argv: Optional[list] = None) -> int:  # pragma: no cover - thin CLI
    import argparse
    parser = argparse.ArgumentParser(
        description="Intertextus - the world, its registers, and the EKG")
    parser.add_argument("command", nargs="?", default="state",
                        choices=["state", "whisper", "ekg"])
    parser.add_argument("--last", type=int, default=10,
                        help="ekg: how many pulses to show")
    args = parser.parse_args(argv)
    if args.command == "whisper":
        print(whisper())
    elif args.command == "ekg":
        path = _heartbeat_path()
        if not path.is_file():
            print("no pulses yet - the world turns when HERMES_SPIRIT=1")
            return 0
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[-max(1, args.last):]:
            try:
                p = json.loads(line)
            except ValueError:
                continue
            print("%s  %-7s %-15s %s  %s" % (
                p.get("ts", ""), p.get("kind", ""), p.get("event") or "-",
                p.get("place", ""),
                " ".join(f"{k}={v}" for k, v in (p.get("registers") or {}).items())))
    else:
        st = state()
        print(f"world: {WORLD_NAME}   (HERMES_SPIRIT="
              f"{'on' if spirit_enabled() else 'off'})")
        print(whisper(st))
        for category in REGISTER_ORDER:
            print("  %-8s %+d" % (category, st.get("registers", {}).get(category, 0)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
