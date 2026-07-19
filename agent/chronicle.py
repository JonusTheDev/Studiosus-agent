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
