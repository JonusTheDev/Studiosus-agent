"""The Chronicle — what the Soul's episodes *meant*, indexed by situation.

The Soul records what happened.  This records what it was worth carrying, and
lays it back beside the work the next time a task resembles the last one.
Phase B of the Studiosus graft (``.plans/studiosus-heart-strategy.md``); it
stands on Phase A and nothing else.

The loop is small:

  1. A turn seals an episode (``agent/soul.py``).
  2. In the quiet afterwards, an auxiliary model reads the raw record and
     distills **at most three** durable lessons: *when <situation>: <what
     worked or failed, and why>*, each tagged.
  3. At the start of a resembling task, the highest-scoring lessons are laid
     beside the work, and the ones actually served are **reinforced**.

Covenants carried from the ancestor Loom, distilled rather than ported:

* **Reflection is free growth.**  A lesson is understanding, and understanding
  harms no one, so writing one is ungated.  What is gated is *acting* on it.
* **An honest empty reflection beats a manufactured lesson.**  A turn that
  taught nothing shelves nothing.
* **The past keeps its one meaning.**  Lessons are keyed to the episode that
  bore them, so re-reflecting on an episode cannot mint duplicates.
* **Reinforced is never shed.**  Consolidation is Phase C's work, but the
  count it will depend on is kept from the first day, because it cannot be
  reconstructed later.
* **Never the main prompt cache.**  Distillation runs through
  ``agent.auxiliary_client.call_llm``, which is a separate call on a separate
  client — a Hermes invariant, and the reason reflection can be slow without
  costing the user anything.

Lessons live at ``$HERMES_HOME/chronicle/lessons.jsonl`` — durable user data
beside skills and memories, not a cache.  The raw Soul may age out; the
distilled must not.

Off by default: ``HERMES_CHRONICLE=1``.  Requires the Soul, since there is
nothing to reflect on without it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from agent import soul
from hermes_constants import get_hermes_home
from utils import is_truthy_value


def _log():
    import logging
    return logging.getLogger(__name__)

MAX_LESSONS_PER_EPISODE = 3  # distillation, not transcription
MAX_LESSON_CHARS = 600       # a lesson is a card, not an essay
SERVE_K = 3                  # how many lessons may be laid beside one task

# Reflection runs in the background after the answer is delivered, so it can
# afford to be slow - and on a partly-offloaded local model it will be. The
# interactive default timed out against qwen3.6 on this box (measured), then
# the auxiliary client's own fallback returned to the misconfigured model and
# failed for a second, unrelated reason. Nobody is waiting on this call.
REFLECT_TIMEOUT_S = 600.0

# Scoring weights.  Tags weigh heaviest so "what did I learn doing X?" surfaces
# the right scars even when the wording of the task has changed completely.
TAG_WEIGHT = 3.0
KEYWORD_WEIGHT = 2.0
TITLE_WEIGHT = 1.0

_WORD = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset("""
a an and are as at be but by for from has have how i if in is it its of on or
que that the then there these they this to was were what when where which
while who why will with you your me my our we do does did can could should
would please help need want make made get got use used using
""".split())

REFLECT_PROMPT = """\
You are reflecting on a finished episode of your own work, in the quiet after \
the toil. Below is the raw record of what you thought, called, got back, and \
delivered.

Distill AT MOST {max} durable lessons worth carrying to future tasks that \
RESEMBLE this one. A lesson names the situation, what worked or failed, and \
why - concrete, not platitude. "Be careful with files" teaches nothing; \
"when editing a file whose path contains spaces, quote the path or the tool \
reports success while writing nothing" teaches something.

If the episode taught nothing new, return an empty list. An honest empty \
reflection is worth more than a manufactured lesson.

EPISODE RECORD:
{episode}

Reply with ONLY a JSON object, no prose around it:
{{"lessons": [{{"title": "...", "tags": ["..."], "keywords": ["..."],
   "text": "when <situation>: <what worked or failed, and why>"}}]}}
"""


def chronicle_enabled() -> bool:
    """True when turns should reflect.  Off unless ``HERMES_CHRONICLE``."""
    return is_truthy_value(os.environ.get("HERMES_CHRONICLE"))


def chronicle_dir() -> Path:
    return get_hermes_home() / "chronicle"


def _lessons_path() -> Path:
    return chronicle_dir() / "lessons.jsonl"


def _reinforcement_path() -> Path:
    return chronicle_dir() / "reinforcement.json"


def _words(text: Any) -> set[str]:
    return {w for w in _WORD.findall(str(text).lower())
            if len(w) > 2 and w not in _STOPWORDS}


# --- the shelf -----------------------------------------------------------


def entries() -> list[dict]:
    """Every shelved lesson, oldest first.  A corrupt line is skipped, never fatal."""
    path = _lessons_path()
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


def add(lesson: dict) -> bool:
    """Shelve one lesson.  False if its id is already present.

    Duplicate ids are how "the past keeps its one meaning" is enforced:
    reflecting twice on one episode must not double its lessons.
    """
    if any(e.get("id") == lesson.get("id") for e in entries()):
        return False
    chronicle_dir().mkdir(parents=True, exist_ok=True)
    with open(_lessons_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(lesson, ensure_ascii=False) + "\n")
    return True


def reinforcement() -> dict:
    """How often each lesson has been laid beside real work."""
    try:
        return json.loads(_reinforcement_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def reinforce(lesson_ids: Iterable[str]) -> None:
    """Strengthen the threads actually used.  Called when lessons are served.

    Phase C may only shed what was never reinforced, and a count not kept from
    the beginning cannot be reconstructed afterwards — which is why this exists
    before anything reads it.
    """
    lesson_ids = [lid for lid in lesson_ids if lid]
    if not lesson_ids:
        return
    counts = reinforcement()
    for lid in lesson_ids:
        counts[lid] = counts.get(lid, 0) + 1
    chronicle_dir().mkdir(parents=True, exist_ok=True)
    _reinforcement_path().write_text(
        json.dumps(counts, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# --- laying lessons beside the work --------------------------------------


def score(lesson: dict, task_words: set[str], task_tags: Iterable[str] = ()) -> float:
    """How well one lesson matches the situation at hand.

    Tags weigh heaviest, then keywords, then the title's own words.  The
    lesson body is deliberately NOT scored: it is a sentence about a
    situation, and matching on its prose surfaces lessons that merely share
    common words with the task.
    """
    task_tags = {str(t).lower() for t in task_tags}
    tags = {str(t).lower() for t in lesson.get("tags", [])}
    keywords = {str(k).lower() for k in lesson.get("keywords", [])}

    hits = len(tags & task_tags) + len(tags & task_words)
    total = TAG_WEIGHT * hits
    total += KEYWORD_WEIGHT * len(keywords & task_words)
    total += TITLE_WEIGHT * len(_words(lesson.get("title", "")) & task_words)
    return total


def retrieve(task_text: str, *, tags: Iterable[str] = (), k: int = SERVE_K) -> list[dict]:
    """The lessons most worth laying beside this task, best first.

    Returns [] when nothing scores above zero — an irrelevant lesson in the
    context window is worse than none, because it spends the model's attention
    to say something about a different situation.
    """
    task_words = _words(task_text)
    scored = [(score(lesson, task_words, tags), lesson) for lesson in entries()]
    hits = [(s, lesson) for s, lesson in scored if s > 0]
    counts = reinforcement()
    # Reinforcement breaks ties only: a lesson that has helped before is more
    # likely to help again, but it must never outrank a better-matching one.
    hits.sort(key=lambda pair: (pair[0], counts.get(pair[1].get("id", ""), 0)),
              reverse=True)
    return [lesson for _, lesson in hits[:max(0, k)]]


def render(lessons: list[dict]) -> str:
    """The block laid into the prompt.  Empty string for no lessons."""
    if not lessons:
        return ""
    lines = ["Lessons from your own past work on tasks like this one:"]
    for lesson in lessons:
        lines.append(f"- {lesson['text']}")
    return "\n".join(lines)


# --- reflection ----------------------------------------------------------


def _digest(events: list[dict], max_chars: int = 6000) -> str:
    """One line per event — the episode as the reflector sees it."""
    lines = []
    for e in events:
        body = {k: v for k, v in e.items() if k not in ("seq", "ts", "type")}
        rendered = "; ".join(f"{k}={str(v)[:200]}" for k, v in body.items())
        lines.append(f"{e.get('seq', 0):02d} {e['type']}: {rendered}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[digest truncated]"
    return text


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Small models fence their JSON, prefix it with prose, or think out loud
    first.  A reply we cannot parse shelves nothing and raises nothing.
    """
    if not text:
        return {}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return {}


def reflect(episode_id: str, call_fn: Optional[Callable] = None,
            *, extra_tags: Iterable[str] = (),
            runtime: Optional[dict] = None) -> list[dict]:
    """Distill one sealed episode into lessons.  Returns what was shelved.

    ``call_fn(prompt) -> str`` is injected so the whole path is testable
    without a model; the default routes through the auxiliary client, never
    the main session's client.  ``runtime`` describes the live session's
    provider/model so reflection can fall back to it — see
    :func:`_auxiliary_call`.

    Never raises.  A failed reflection is a quiet evening, not a broken
    harness — the turn it reflects on has already been delivered.  It is a
    *logged* quiet evening, though: a reflection that silently produces
    nothing is indistinguishable from a misconfiguration, and that cost us an
    hour once.
    """
    try:
        events = soul.read_episode(episode_id)
    except (FileNotFoundError, OSError, ValueError):
        return []
    if not events or events[-1].get("type") != soul.EPISODE_END:
        return []  # an unsealed episode is not yet a memory

    prompt = REFLECT_PROMPT.format(max=MAX_LESSONS_PER_EPISODE,
                                   episode=_digest(events))
    try:
        reply = (call_fn(prompt) if call_fn
                 else _auxiliary_call(prompt, runtime=runtime))
    except Exception as exc:
        _log().warning("chronicle: reflection call failed for %s: %s",
                       episode_id, exc)
        return []

    lessons = _extract_json(reply or "").get("lessons")
    if not isinstance(lessons, list):
        return []

    outcome = events[-1].get("outcome")
    shelved = []
    for n, lesson in enumerate(lessons[:MAX_LESSONS_PER_EPISODE], start=1):
        if not isinstance(lesson, dict) or not str(lesson.get("text", "")).strip():
            continue
        item = {
            "id": f"{episode_id}-L{n}",
            "title": str(lesson.get("title", "untitled lesson"))[:120],
            "text": str(lesson["text"])[:MAX_LESSON_CHARS],
            "tags": [str(t)[:40] for t in lesson.get("tags", [])
                     if isinstance(lesson.get("tags"), list)]
                    + [str(t) for t in extra_tags if t],
            "keywords": [str(k)[:40] for k in lesson.get("keywords", [])
                         if isinstance(lesson.get("keywords"), list)],
            "source_episode": episode_id,
            "source_outcome": outcome,
            "created": soul.now_ts(),
        }
        if add(item):
            shelved.append(item)
    if shelved:
        try:
            from agent.spirit import beat as _spirit_beat
            _spirit_beat("lesson_earned", episode_id=episode_id)
        except Exception:
            pass  # reflection sharpens whether or not the world turns
    return shelved


def spawn_reflection(episode_id: str, *, extra_tags: Iterable[str] = (),
                     runtime: Optional[dict] = None) -> None:
    """Reflect on a sealed episode in the background.

    The turn it reflects on has already been delivered, so this must never
    make the user wait and must never be able to fail loudly.  A daemon thread
    means a reflection still in flight cannot hold the process open at exit —
    an unfinished lesson is a small loss, a hung shutdown is a large one.
    """
    import threading

    def _run() -> None:
        try:
            shelved = reflect(episode_id, extra_tags=extra_tags, runtime=runtime)
            _log().info("chronicle: reflection on %s shelved %d lesson(s)",
                        episode_id, len(shelved))
        except Exception as exc:
            # A quiet evening, not a broken harness - but never a silent one.
            _log().warning("chronicle: reflection on %s failed: %s",
                           episode_id, exc)

    try:
        threading.Thread(target=_run, name="chronicle-reflect", daemon=True).start()
    except Exception:
        pass


def _auxiliary_call(prompt: str, runtime: Optional[dict] = None) -> str:
    """Route reflection through the auxiliary client.

    Never the main session's *client*: a Hermes invariant, and the reason a
    slow reflection costs the user nothing.  Configure a cheap model under
    ``auxiliary.reflection`` in config.yaml and it will be used.

    When that resolves to nothing usable, fall back to the model that did the
    work.  Measured need: auxiliary auto-detect picked this box's configured
    default (``gemma3:4b``) which was not actually installed, so every
    reflection failed while the model that had just completed the turn sat
    loaded and idle.  Reflecting with the main runtime through the auxiliary
    client is still a separate call on a separate client — the prompt cache
    is untouched — so the invariant holds and the feature works out of the box.
    """
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    messages = [{"role": "user", "content": prompt}]
    try:
        response = call_llm(task="reflection", messages=messages,
                            temperature=0.3, max_tokens=800,
                            timeout=REFLECT_TIMEOUT_S)
        return extract_content_or_reasoning(response) or ""
    except Exception as exc:
        if not runtime or not runtime.get("model"):
            raise
        _log().info(
            "chronicle: configured reflection model unusable (%s); "
            "falling back to the session's own model %s", exc, runtime["model"])

    response = call_llm(
        messages=messages, temperature=0.3, max_tokens=800,
        timeout=REFLECT_TIMEOUT_S,
        model=runtime.get("model"), provider=runtime.get("provider") or None,
        base_url=runtime.get("base_url") or None,
        api_key=runtime.get("api_key") or None,
    )
    return extract_content_or_reasoning(response) or ""


# ---------------------------------------------------------------------------
# Phase C - consolidation: the distillation of the distilled.
#
# "A dreamy night's sleep": kin lessons are merged into sharper ones and
# platitudes released, so the shelf holds more meaning than the day before
# while growing smaller.  The covenant, enforced in apply_consolidation()
# REGARDLESS of what the dream asked:
#
#   * a lesson with reinforcement > 0 is NEVER shed - it is load-bearing;
#     it may still be merged, and the merge inherits its strength;
#   * shed and absorbed lessons are archived to shed.jsonl, never burned;
#   * propose-then-apply: the dream writes a proposal file and changes
#     NOTHING; only apply_consolidation - a human's deliberate act, via the
#     CLI - touches the shelf, and a proposal applies at most once.
# ---------------------------------------------------------------------------

BATCH_SIZE = 10  # a dream works one small neighborhood of kin at a time; the
# ancestor measured whole-shelf dreams overflowing context and timing out,
# and sorting by tags lands kin together so small batches still find them.

CONSOLIDATE_PROMPT = """\
You are consolidating your own chronicle of lessons in the quiet of sleep - \
merging kin, releasing what never mattered. Below are your lessons, each with \
an id and a REINFORCED count (how often it was laid beside real work).

Rules:
- MERGE lessons that say the same thing in different coats: write ONE sharper \
lesson and list the absorbed ids. The merged text must preserve every \
distinct situation the absorbed lessons covered.
- SHED only lessons that are empty platitudes teaching nothing situational. A \
lesson with REINFORCED > 0 must NEVER be shed (it is load-bearing; you may \
still merge it). When unsure, keep. An untouched lesson needs no mention.

LESSONS:
{lessons}

Reply with ONLY a JSON object, no prose around it:
{{"merges": [{{"title": "...", "tags": ["..."], "keywords": ["..."],
    "text": "when <situation>: <the sharpened lesson>", "absorb": ["id", "id"]}}],
  "shed": ["id"]}}
"""


def consolidation_dir() -> Path:
    return chronicle_dir() / "consolidation"


def _shed_path() -> Path:
    return chronicle_dir() / "shed.jsonl"


def _rewrite(lessons: list[dict]) -> None:
    """Replace the whole shelf.  Consolidation is the ONLY caller: everything
    else appends, because the past is not edited."""
    chronicle_dir().mkdir(parents=True, exist_ok=True)
    with open(_lessons_path(), "w", encoding="utf-8") as f:
        for lesson in lessons:
            f.write(json.dumps(lesson, ensure_ascii=False) + "\n")


def _archive_shed(lessons: list[dict], reason: str) -> None:
    """Released, not burned.  A shed lesson can always be read back."""
    if not lessons:
        return
    chronicle_dir().mkdir(parents=True, exist_ok=True)
    with open(_shed_path(), "a", encoding="utf-8") as f:
        for lesson in lessons:
            f.write(json.dumps({**lesson, "shed_reason": reason,
                                "shed_at": soul.now_ts()},
                               ensure_ascii=False) + "\n")


def propose_consolidation(call_fn: Optional[Callable] = None, *,
                          write: bool = True, batch_size: int = BATCH_SIZE,
                          runtime: Optional[dict] = None) -> dict:
    """Dream up a consolidation.  Applies NOTHING.

    Lessons are sorted by tags so kin cluster into the same batch, and each
    batch is dreamt over independently - a malformed reply loses one batch's
    suggestions, never the run.  Returns ``{"proposal", "path"}``.
    """
    lessons = sorted(entries(),
                     key=lambda e: (",".join(e.get("tags", [])), e.get("title", "")))
    counts = reinforcement()
    merges: list[dict] = []
    shed: list[str] = []
    for i in range(0, len(lessons), batch_size):
        batch = lessons[i:i + batch_size]
        lines = ["- id=%s REINFORCED=%d [%s] %s :: %s"
                 % (lesson["id"], counts.get(lesson["id"], 0),
                    ", ".join(lesson.get("tags", [])),
                    lesson.get("title", ""), lesson.get("text", "")[:200])
                 for lesson in batch]
        try:
            prompt = CONSOLIDATE_PROMPT.format(lessons="\n".join(lines))
            reply = call_fn(prompt) if call_fn else _auxiliary_call(prompt,
                                                                    runtime=runtime)
        except Exception as exc:
            _log().warning("chronicle: consolidation batch %d failed: %s",
                           i // batch_size + 1, exc)
            continue
        raw = _extract_json(reply or "")
        merges += [m for m in (raw.get("merges") or []) if isinstance(m, dict)]
        shed += [sid for sid in (raw.get("shed") or []) if isinstance(sid, str)]

    proposal = {"proposed_at": soul.now_ts(), "lesson_count": len(lessons),
                "merges": merges, "shed": shed}
    path = None
    if write:
        consolidation_dir().mkdir(parents=True, exist_ok=True)
        path = consolidation_dir() / ("proposal-%s.json" % proposal["proposed_at"])
        path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return {"proposal": proposal, "path": str(path) if path else None}


def apply_consolidation(proposal: dict) -> dict:
    """Apply a reviewed proposal.  The covenant is enforced HERE, regardless
    of what the dream asked - the dream is counsel, this is law.

    Returns ``{"before", "after", "merged", "released", "refused"}`` where
    each refusal names the id and the reason, so the reviewing hand sees
    exactly what was declined and why.
    """
    shelf = {lesson["id"]: lesson for lesson in entries()}
    counts = reinforcement()
    ts = proposal.get("proposed_at", soul.now_ts())

    refused: list[dict] = []
    removed: set[str] = set()

    for lid in proposal.get("shed", []):
        if lid not in shelf:
            refused.append({"id": lid, "why": "unknown id"})
        elif counts.get(lid, 0) > 0:
            refused.append({"id": lid, "why": "reinforced; never shed"})
        else:
            removed.add(lid)

    merged: list[dict] = []
    for n, m in enumerate(proposal.get("merges", []), start=1):
        absorb = [a for a in m.get("absorb", []) if a in shelf and a not in removed]
        if len(absorb) < 2 or not str(m.get("text", "")).strip():
            refused.append({"id": m.get("title", "merge-%d" % n),
                            "why": "merge needs >=2 known absorbed ids and a text"})
            continue
        # The merged text must preserve every situation its kin covered - and
        # tags are how a situation is FOUND, so the kin's tags are unioned in
        # regardless of what the dream wrote. A merge that lost its parents'
        # tags would preserve the words while orphaning the retrieval.
        dream_tags = [str(t)[:40] for t in m.get("tags", [])
                      if isinstance(m.get("tags"), list)]
        kin_tags = [t for a in absorb for t in shelf[a].get("tags", [])]
        merged.append({
            "id": "consolidated-%s-%d" % (ts, n),
            "title": str(m.get("title", "consolidated lesson"))[:120],
            "text": str(m["text"])[:MAX_LESSON_CHARS],
            "tags": list(dict.fromkeys(dream_tags + kin_tags)),
            "keywords": [str(k)[:40] for k in m.get("keywords", [])
                         if isinstance(m.get("keywords"), list)],
            "source_episode": None,
            "source_outcome": None,
            "absorbed": absorb,
            "created": soul.now_ts(),
        })
        removed.update(absorb)

    released = [shelf[lid] for lid in removed]
    kept = [lesson for lesson in entries() if lesson["id"] not in removed]
    _archive_shed(released, reason="consolidation:%s" % ts)
    _rewrite(kept + merged)

    # A merge inherits the strength of what it absorbed: consolidation must
    # never be a way to launder a load-bearing lesson into a shed-able one.
    if merged:
        new_counts = reinforcement()
        for item in merged:
            inherited = sum(counts.get(a, 0) for a in item["absorbed"])
            if inherited:
                new_counts[item["id"]] = new_counts.get(item["id"], 0) + inherited
        chronicle_dir().mkdir(parents=True, exist_ok=True)
        _reinforcement_path().write_text(
            json.dumps(new_counts, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")

    if merged or released:
        try:
            from agent.spirit import beat as _spirit_beat
            _spirit_beat("rested")  # a dreamy night's sleep, truly slept
        except Exception:
            pass

    return {"before": len(shelf), "after": len(kept) + len(merged),
            "merged": len(merged), "released": len(released), "refused": refused}


def list_proposals() -> list[dict]:
    """Every proposal on the desk, oldest first.  Malformed files are skipped."""
    directory = consolidation_dir()
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("proposal-*.json")):
        try:
            prop = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"ts": prop.get("proposed_at", path.stem[len("proposal-"):]),
                    "merges": len(prop.get("merges", [])),
                    "shed": len(prop.get("shed", [])),
                    "applied_at": prop.get("applied_at"),
                    "path": str(path)})
    return out


def apply_proposal_file(ts: str) -> dict:
    """Apply one desk proposal by timestamp, exactly once.

    Raises ``KeyError`` for a proposal that does not exist and ``ValueError``
    for one already applied - a proposal is a moment's counsel, and the shelf
    it described no longer exists after the first application.
    """
    path = consolidation_dir() / ("proposal-%s.json" % ts)
    if not path.is_file():
        raise KeyError("no proposal %s on the desk" % ts)
    proposal = json.loads(path.read_text(encoding="utf-8"))
    if proposal.get("applied_at"):
        raise ValueError("proposal %s was already applied at %s"
                         % (ts, proposal["applied_at"]))
    summary = apply_consolidation(proposal)
    proposal["applied_at"] = soul.now_ts()
    proposal["applied_summary"] = summary
    path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return summary


def _main(argv: Optional[list] = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agent.chronicle",
        description="The lesson shelf: inspect it, dream a consolidation, apply one.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("shelf", help="list every lesson with its reinforcement")
    sub.add_parser("desk", help="list consolidation proposals")
    p_prop = sub.add_parser("propose",
                            help="dream a consolidation proposal (changes nothing)")
    p_prop.add_argument("--model", default=None,
                        help="model to dream with when auxiliary.reflection is "
                             "unusable, e.g. qwen3.6:latest")
    p_prop.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    p_prop.add_argument("--provider", default="custom")
    p_prop.add_argument("--api-key", default="ollama")
    p_apply = sub.add_parser("apply", help="apply one reviewed proposal")
    p_apply.add_argument("ts", help="proposal timestamp, from 'desk'")
    args = parser.parse_args(argv)

    if args.cmd == "shelf":
        counts = reinforcement()
        for lesson in entries():
            print("%-46s r=%-3d [%s] %s" % (lesson["id"][:46],
                  counts.get(lesson["id"], 0),
                  ", ".join(lesson.get("tags", [])), lesson.get("title", "")))
        print("%d lesson(s); shed archive: %s" % (
            len(entries()), _shed_path() if _shed_path().is_file() else "(empty)"))
    elif args.cmd == "desk":
        proposals = list_proposals()
        for prop in proposals:
            state = ("applied %s" % prop["applied_at"]) if prop["applied_at"] else "PENDING"
            print("%s  merges=%d shed=%d  %s" % (prop["ts"], prop["merges"],
                                                 prop["shed"], state))
        if not proposals:
            print("the desk is empty - dream one with: propose")
    elif args.cmd == "propose":
        runtime = ({"model": args.model, "provider": args.provider,
                    "base_url": args.base_url, "api_key": args.api_key}
                   if args.model else None)
        got = propose_consolidation(runtime=runtime)
        prop = got["proposal"]
        print("proposed: %d merge(s), %d shed over %d lesson(s)"
              % (len(prop["merges"]), len(prop["shed"]), prop["lesson_count"]))
        print("written to %s" % got["path"])
        print("review it, then: python -m agent.chronicle apply %s" % prop["proposed_at"])
    elif args.cmd == "apply":
        summary = apply_proposal_file(args.ts)
        print("shelf: %d -> %d (%d merged, %d released to the shed archive)"
              % (summary["before"], summary["after"], summary["merged"],
                 summary["released"]))
        for r in summary["refused"]:
            print("  refused %s: %s" % (r["id"], r["why"]))
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
