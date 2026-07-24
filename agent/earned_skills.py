"""Earned skills — playbooks distilled from observed repetition, never invented.

Phase D of the Studiosus graft (``.plans/studiosus-heart-strategy.md``), and
its sharpest covenant: **a skill is born only of repetition**.  The single
door is the distill dream below — kin lessons are gathered into families by
tag, and only a family drawing on at least ``MIN_FAMILY`` distinct episodes
that verifiably sealed ``complete`` may be offered for drafting.  Failed toil
teaches lessons; only what repeatedly *worked* may become a playbook.

A skill here is not machinery.  It never executes and never composes: it is
words laid beside the work — a name, one line naming when to reach for it, a
few ordered steps — advisory, and earned.  (Hermes' own ``SKILL.md`` system
is machinery; rerouting its agent-created path through this covenant is later
work.  This shelf stands beside it, not inside it.)

Covenants, distilled from the ancestor Loom:

* **Strictness over convenience.**  A family counts only episodes whose
  ``complete`` seal can be re-verified against the Soul right now.  A lesson
  whose episode aged out lends its words to a draft, never its episode count.
* **Propose-then-apply, per draft.**  The dream writes a proposal and changes
  nothing.  Each draft is resolved by a reviewing hand — ``shelved`` (through
  the full apply covenant; a refusal leaves it pending) or ``set_aside`` —
  and a resolved draft is never re-dreamt.
* **A card, not a book.**  A playbook over ``MAX_SKILL_CHARS`` is refused,
  not trimmed — the reviewing hand edits and applies again; no silent scissor.
* **Usage is not effectiveness.**  ``correlation_report`` ties served-skill →
  sealed-episode outcome, worst-rate-first, so the skills most worth scrutiny
  surface at the top.  A never-served skill reports no data, not zero —
  silence is not failure.

Storage sits beside the Chronicle's: ``$HERMES_HOME/chronicle/skills/``.
Serving rides the same ``HERMES_CHRONICLE`` flag — the heart is one organ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

from agent import chronicle, soul


def _log():
    import logging
    return logging.getLogger(__name__)


MIN_FAMILY = 3        # a skill is born of repetition, and three is the least of it
MAX_SKILL_CHARS = 1200  # a playbook is a card laid on the bench, not a book
SERVE_K_SKILLS = 2    # at most this many playbooks laid beside one task

DISTILL_PROMPT = """\
You are distilling a SKILL - a short playbook of what has repeatedly worked in \
your own past toil. Below is one family of kin lessons, all wearing the tag \
"{tag}", drawn from {n} of your episodes that truly ended complete.

A skill is words laid on the bench beside your tools: a name, one line naming \
when to reach for it, and a few ordered steps in your own voice. It must \
describe only what these lessons actually bear out - nothing invented, nothing \
aspirational. If this family does not truly form one repeatable playbook, say \
so honestly: an empty hand is worth more than a false card.

LESSONS OF THIS FAMILY:
{lessons}

Reply with ONLY a JSON object, no prose around it:
{{"skill": {{"title": "...", "reach_for_when": "one plain line",
   "steps": ["first ...", "then ..."], "tags": ["..."], "keywords": ["..."]}}}}
or, if no honest playbook lives here: {{"skill": null}}
"""


def skills_dir() -> Path:
    return chronicle.chronicle_dir() / "skills"


def _entries_path() -> Path:
    return skills_dir() / "entries.jsonl"


def _reinforcement_path() -> Path:
    return skills_dir() / "reinforcement.json"


def proposals_dir() -> Path:
    return skills_dir() / "proposals"


# --- the shelf (same card-catalogue shape as the Chronicle) ---------------


def entries() -> list[dict]:
    """Every shelved skill, oldest first.  A corrupt line is skipped."""
    path = _entries_path()
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


def add(skill: dict) -> bool:
    """Shelve one skill.  False if its id is already present."""
    if any(e.get("id") == skill.get("id") for e in entries()):
        return False
    skills_dir().mkdir(parents=True, exist_ok=True)
    with open(_entries_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(skill, ensure_ascii=False) + "\n")
    return True


def reinforcement() -> dict:
    try:
        return json.loads(_reinforcement_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def reinforce(skill_ids: Iterable[str]) -> None:
    """A playbook laid beside real work is a thread strengthened, same as a
    lesson — and a reinforced skill may never be shed, only mended."""
    skill_ids = [sid for sid in skill_ids if sid]
    if not skill_ids:
        return
    counts = reinforcement()
    for sid in skill_ids:
        counts[sid] = counts.get(sid, 0) + 1
    skills_dir().mkdir(parents=True, exist_ok=True)
    _reinforcement_path().write_text(
        json.dumps(counts, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def retrieve(task_text: str, *, k: int = SERVE_K_SKILLS) -> list[dict]:
    """The playbooks most worth laying beside this task, best first.

    Skills share the Chronicle's scoring (tags heaviest) and its silence rule:
    nothing scores above zero, nothing is served.
    """
    task_words = chronicle._words(task_text)
    scored = [(chronicle.score(skill, task_words), skill) for skill in entries()]
    hits = [(s, skill) for s, skill in scored if s > 0]
    counts = reinforcement()
    hits.sort(key=lambda pair: (pair[0], counts.get(pair[1].get("id", ""), 0)),
              reverse=True)
    return [skill for _, skill in hits[:max(0, k)]]


def render(skills: list[dict]) -> str:
    """The block laid into the prompt.  Empty string for no skills."""
    if not skills:
        return ""
    lines = ["Playbooks earned from your own repeated successes:"]
    for skill in skills:
        lines.append(f"## {skill['title']}\n{skill['text']}")
    return "\n".join(lines)


# --- families: the only door ----------------------------------------------


def _episode_outcome(episode_id: Optional[str]) -> Optional[str]:
    """The sealed outcome of an episode, re-verified against the Soul now.

    None for an unsealed, missing, or aged-out episode — an episode that
    cannot be verified does not count toward a family.  Strictness over
    convenience: this is the data-quality covenant.
    """
    if not episode_id:
        return None
    try:
        events = soul.read_episode(episode_id)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if events and events[-1].get("type") == soul.EPISODE_END:
        return events[-1].get("outcome")
    return None


def gather_families(min_family: int = MIN_FAMILY) -> list[dict]:
    """Group the Chronicle's lessons into kin families by tag.

    A family is fit to become a skill only when it draws on at least
    ``min_family`` DISTINCT source episodes that verifiably sealed
    ``complete``.  Tags already worn by a shelved skill are left alone — a
    craft is not re-dreamt — and consolidated lessons lend their words to a
    family without lending it an episode count (their provenance is folded
    away).  Returns families largest-first: {tag, lessons, episodes}.
    """
    worn: set[str] = set()
    for skill in entries():
        worn.update(skill.get("tags", []))

    by_tag: dict[str, dict] = {}
    for lesson in chronicle.entries():
        episode_id = lesson.get("source_episode")
        for tag in lesson.get("tags", []):
            if tag in worn:
                continue
            family = by_tag.setdefault(tag, {"tag": tag, "lessons": [],
                                             "episodes": set()})
            family["lessons"].append(lesson)
            if episode_id and _episode_outcome(episode_id) == soul.OUTCOME_COMPLETE:
                family["episodes"].add(episode_id)

    families = [{**f, "episodes": sorted(f["episodes"])}
                for f in by_tag.values() if len(f["episodes"]) >= min_family]
    return sorted(families, key=lambda f: (-len(f["episodes"]), f["tag"]))


def propose_skills(call_fn: Optional[Callable] = None, *,
                   runtime: Optional[dict] = None, write: bool = True,
                   min_family: int = MIN_FAMILY,
                   max_families: Optional[int] = None,
                   skip_tags: Iterable[str] = ()) -> dict:
    """Dream skill drafts from the gathered families.  Applies NOTHING.

    A family the dream judges to hold no honest playbook yields no draft.
    ``skip_tags`` leaves named families alone (a family already waiting in a
    pending draft is not re-dreamt); ``max_families`` caps one dream's reach.
    """
    families = gather_families(min_family=min_family)
    if skip_tags:
        families = [f for f in families if f["tag"] not in set(skip_tags)]
    if max_families is not None:
        families = families[:max_families]

    drafts = []
    for family in families:
        lines = ["- [%s] %s :: %s" % (", ".join(lesson.get("tags", [])),
                                      lesson.get("title", ""),
                                      lesson.get("text", "")[:300])
                 for lesson in family["lessons"]]
        prompt = DISTILL_PROMPT.format(tag=family["tag"],
                                       n=len(family["episodes"]),
                                       lessons="\n".join(lines))
        try:
            reply = (call_fn(prompt) if call_fn
                     else chronicle._auxiliary_call(prompt, runtime=runtime))
        except Exception as exc:
            _log().warning("earned_skills: distill of family %r failed: %s",
                           family["tag"], exc)
            continue
        sk = chronicle._extract_json(reply or "").get("skill")
        if isinstance(sk, dict) and sk.get("title") and sk.get("steps"):
            drafts.append({"family_tag": family["tag"],
                           "episodes": family["episodes"], "skill": sk})
            _log().info("earned_skills: family %r (%d episodes): drafted %r",
                        family["tag"], len(family["episodes"]), sk["title"])
        else:
            _log().info("earned_skills: family %r: no honest playbook found - "
                        "left alone", family["tag"])

    proposal = {"proposed_at": soul.now_ts(), "family_count": len(families),
                "drafts": drafts}
    path = None
    if write:
        proposals_dir().mkdir(parents=True, exist_ok=True)
        path = proposals_dir() / ("proposal-%s.json" % proposal["proposed_at"])
        path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return {"proposal": proposal, "path": str(path) if path else None}


# --- apply: the covenant, enforced whatever the dream asked ---------------


def _render_card(sk: dict) -> str:
    steps = "\n".join("%d. %s" % (i, s)
                      for i, s in enumerate(sk.get("steps", []), start=1))
    return "Reach for this when: %s\nSteps:\n%s" % (
        str(sk.get("reach_for_when", "")).strip(), steps)


def apply_skills(proposal: dict, indices: Optional[Iterable[int]] = None) -> dict:
    """Shelve a reviewed proposal's drafts.  Enforced here regardless of what
    the dream asked: a draft needs a title, a reach-for line, non-empty steps,
    and provenance of >= MIN_FAMILY episodes; a card over MAX_SKILL_CHARS is
    refused, not trimmed.  Duplicate ids are refused by the shelf itself.

    ``indices`` (0-based) shelves only the named drafts — the reviewing hand
    may bless one card at a time.  Skill ids keep their position in the FULL
    proposal, so blessing draft 3 today and draft 1 tomorrow can never mint
    the same id twice.  Returns {"shelved", "refused"}.
    """
    indices = set(indices) if indices is not None else None
    ts = proposal.get("proposed_at", soul.now_ts())
    shelved, refused = [], []
    for n, draft in enumerate(proposal.get("drafts", []), start=1):
        if indices is not None and (n - 1) not in indices:
            continue
        sk = draft.get("skill") or {}
        steps = sk.get("steps")
        if not (sk.get("title") and sk.get("reach_for_when")
                and isinstance(steps, list) and steps):
            refused.append({"id": sk.get("title", "draft-%d" % n),
                            "why": "needs title, reach_for_when, and steps"})
            continue
        if len(draft.get("episodes", [])) < MIN_FAMILY:
            refused.append({"id": sk["title"],
                            "why": "provenance below %d episodes" % MIN_FAMILY})
            continue
        text = _render_card(sk)
        if len(text) > MAX_SKILL_CHARS:
            refused.append({"id": sk["title"],
                            "why": "a playbook is a card, not a book "
                                   "(%d > %d chars)" % (len(text), MAX_SKILL_CHARS)})
            continue
        tags = [str(t)[:40] for t in sk.get("tags", [])
                if isinstance(sk.get("tags"), list)]
        if draft.get("family_tag") and draft["family_tag"] not in tags:
            tags.append(draft["family_tag"])  # the family name keeps kinship findable
        item = {
            "id": "skill-%s-%d" % (ts, n),
            "title": str(sk["title"])[:120],
            "text": text,
            "tags": tags,
            "keywords": [str(k)[:40] for k in sk.get("keywords", [])
                         if isinstance(sk.get("keywords"), list)],
            "from_episodes": list(draft.get("episodes", [])),
            "created": soul.now_ts(),
        }
        if add(item):
            shelved.append(item)
        else:
            refused.append({"id": item["id"], "why": "duplicate id"})
    return {"shelved": shelved, "refused": refused}


# --- the desk: per-draft resolution ---------------------------------------


def list_proposals() -> list[dict]:
    """Every distill proposal on the desk, oldest first, drafts indexed."""
    directory = proposals_dir()
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("proposal-*.json")):
        try:
            prop = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        drafts = [dict(d, index=i) for i, d in enumerate(prop.get("drafts", []))]
        out.append({"ts": prop.get("proposed_at", path.stem[len("proposal-"):]),
                    "drafts": drafts})
    return out


def pending_drafts() -> list[dict]:
    """Drafts still awaiting a hand, across every proposal.  The nightly
    dream skips their families (``skip_tags``) so a waiting family is not
    re-dreamt."""
    return [dict(d, proposal_ts=prop["ts"])
            for prop in list_proposals()
            for d in prop["drafts"] if not d.get("resolution")]


def resolve_draft(ts: str, index: int, resolution: str) -> dict:
    """One hand's word on one draft.

    ``shelved`` truly shelves it through the full apply covenant — a refusal
    there leaves the draft pending and is reported, never swallowed.
    ``set_aside`` only marks it.  Raises KeyError for a draft that does not
    exist, ValueError for one already resolved.
    """
    path = proposals_dir() / ("proposal-%s.json" % ts)
    if not path.is_file():
        raise KeyError("no proposal %s on the desk" % ts)
    proposal = json.loads(path.read_text(encoding="utf-8"))
    drafts = proposal.get("drafts", [])
    if not 0 <= index < len(drafts):
        raise KeyError("no draft %d in proposal %s" % (index, ts))
    if drafts[index].get("resolution"):
        raise ValueError("draft %d already %s"
                         % (index, drafts[index]["resolution"]["resolution"]))

    shelved, refused = [], []
    if resolution == "shelved":
        got = apply_skills(proposal, indices=[index])
        shelved, refused = got["shelved"], got["refused"]
        if not shelved:
            return {"resolution": None, "shelved": [], "refused": refused}
    elif resolution != "set_aside":
        raise ValueError("resolution must be 'shelved' or 'set_aside'")

    drafts[index]["resolution"] = {"resolution": resolution, "ts": soul.now_ts()}
    path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return {"resolution": resolution, "shelved": shelved, "refused": refused}


# --- correlation: usage is not effectiveness ------------------------------


def _served_outcomes(skill_id: str) -> list[str]:
    """Every sealed outcome of an episode that had this skill on its bench.

    Unsealed or unreadable episodes are skipped — an outcome can only speak
    once it is truly known.
    """
    outcomes = []
    for episode_id in soul.list_episodes():
        try:
            events = soul.read_episode(episode_id)
        except (FileNotFoundError, OSError, ValueError):
            continue
        assemble = next((e for e in events if e.get("type") == soul.ASSEMBLE), None)
        if not assemble or skill_id not in (assemble.get("skills") or []):
            continue
        if events[-1].get("type") == soul.EPISODE_END:
            outcomes.append(events[-1].get("outcome"))
    return outcomes


def correlation_report() -> list[dict]:
    """One row per shelved skill, worst-rate-first.

    Raw reinforcement counts how often a skill was LAID on the bench, never
    whether the episodes that leaned on it actually worked.  This derives the
    truer view from records the Soul already keeps.  Read-only: it informs a
    steward's mending judgment and changes nothing on its own.  A never-served
    skill reports served=0, rate=None — no data is not zero, and a skill with
    no evidence must never outrank one known to be struggling.
    """
    counts = reinforcement()
    rows = []
    for skill in entries():
        outcomes = _served_outcomes(skill["id"])
        complete = sum(1 for o in outcomes if o == soul.OUTCOME_COMPLETE)
        rate = (complete / len(outcomes)) if outcomes else None
        rows.append({"id": skill["id"], "title": skill.get("title", ""),
                     "served": len(outcomes), "complete": complete,
                     "rate": rate, "reinforced": counts.get(skill["id"], 0)})
    return sorted(rows, key=lambda r: (r["rate"] if r["rate"] is not None else 2,
                                       -r["served"]))


def _main(argv: Optional[list] = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agent.earned_skills",
        description="Playbooks earned from repetition: families, dreams, the desk.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("shelf", help="list every earned skill")
    sub.add_parser("families", help="which tags could become a skill, and how close")
    p_prop = sub.add_parser("propose", help="dream drafts from ready families")
    p_prop.add_argument("--model", default=None)
    p_prop.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    p_prop.add_argument("--provider", default="custom")
    p_prop.add_argument("--api-key", default="ollama")
    sub.add_parser("desk", help="drafts awaiting a hand")
    p_res = sub.add_parser("resolve", help="bless or set aside one draft")
    p_res.add_argument("ts")
    p_res.add_argument("index", type=int)
    p_res.add_argument("resolution", choices=["shelved", "set_aside"])
    sub.add_parser("correlate", help="served-skill vs sealed-outcome, worst first")
    args = parser.parse_args(argv)

    if args.cmd == "shelf":
        for skill in entries():
            print("%-28s [%s] %s" % (skill["id"], ", ".join(skill.get("tags", [])),
                                     skill.get("title", "")))
        print("%d skill(s)" % len(entries()))
    elif args.cmd == "families":
        ready = gather_families()
        for family in ready:
            print("READY  %-24s %d complete episode(s), %d lesson(s)"
                  % (family["tag"], len(family["episodes"]), len(family["lessons"])))
        if not ready:
            print("no family holds %d verified-complete episodes yet - "
                  "the field needs more toil" % MIN_FAMILY)
    elif args.cmd == "propose":
        runtime = ({"model": args.model, "provider": args.provider,
                    "base_url": args.base_url, "api_key": args.api_key}
                   if args.model else None)
        waiting = {d["family_tag"] for d in pending_drafts()}
        got = propose_skills(runtime=runtime, skip_tags=waiting)
        prop = got["proposal"]
        print("dreamt over %d family(ies): %d draft(s)"
              % (prop["family_count"], len(prop["drafts"])))
        if got["path"]:
            print("written to %s" % got["path"])
    elif args.cmd == "desk":
        pending = pending_drafts()
        for d in pending:
            print("%s #%d  [%s]  %s  (%d episodes)"
                  % (d["proposal_ts"], d["index"], d["family_tag"],
                     (d.get("skill") or {}).get("title", "?"), len(d["episodes"])))
        if not pending:
            print("the desk is empty")
    elif args.cmd == "resolve":
        got = resolve_draft(args.ts, args.index, args.resolution)
        if got["resolution"] is None:
            print("draft left PENDING - the covenant refused it:")
        for r in got["refused"]:
            print("  refused %s: %s" % (r["id"], r["why"]))
        for skill in got["shelved"]:
            print("shelved %s: %s" % (skill["id"], skill["title"]))
        if got["resolution"] == "set_aside":
            print("set aside.")
    elif args.cmd == "correlate":
        for row in correlation_report():
            rate = "no data" if row["rate"] is None else "%.0f%%" % (row["rate"] * 100)
            print("%-28s served=%-3d complete=%-3d rate=%-8s reinforced=%d  %s"
                  % (row["id"], row["served"], row["complete"], rate,
                     row["reinforced"], row["title"]))
        if not entries():
            print("no earned skills yet")
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
