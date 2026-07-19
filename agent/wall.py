"""The Wall — where proven work becomes something to be proud of.

Phase E of the Studiosus graft.  The ancestor's Wall held certificates
earned by seeing a whole course of study through, ring by ring; there was no
certificate for participation.  Hermes has no schoolhouse curriculum, but it
has something just as honest: the serve→outcome correlation the Soul keeps
for every earned skill (Phase D).  So the covenant distills to:

    **A certificate is minted only for an earned skill proven in service** —
    laid beside real work at least :data:`MIN_SERVES` times, with a sealed
    ``complete`` rate of at least :data:`CERT_RATE`.

Minting is deterministic from the ledger — :func:`mint` scans, judges once,
and never judges twice: a certificate, once on the wall, stays there even if
the skill later struggles (the correlation report is where ongoing scrutiny
lives; the Wall records what *was* truly earned).  Certificates are never
pruned and there is no delete.

At ASSEMBLE at most one matching certificate is laid beside the work — a
confidence signal, not a playbook: "this ground is earned, not guessed."
It is not reinforced and carries no learning weight.

Storage: ``$HERMES_HOME/wall/entries.jsonl``, kin to the Chronicle's shelf.
Inert by default — certificates only exist once ``python -m agent.wall
mint`` has found something truly earned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from agent import chronicle, earned_skills, soul
from hermes_constants import get_hermes_home

MIN_SERVES = 5     # proven means repeatedly on the bench, not once lucky
CERT_RATE = 0.8    # and the work it stood beside almost always sealed complete
SERVE_K_CERTS = 1  # a confidence signal, not a second shelf of playbooks


def wall_dir() -> Path:
    return get_hermes_home() / "wall"


def _entries_path() -> Path:
    return wall_dir() / "entries.jsonl"


def entries() -> list[dict]:
    """Every certificate on the wall, oldest first.  A corrupt line is
    skipped."""
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


def _add(cert: dict) -> bool:
    if any(e.get("id") == cert.get("id") for e in entries()):
        return False
    wall_dir().mkdir(parents=True, exist_ok=True)
    with open(_entries_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(cert, ensure_ascii=False) + "\n")
    return True


def eligible() -> list[dict]:
    """Correlation rows that meet the covenant and hold no certificate yet.

    A never-served skill (rate None) is never eligible: no data is not
    proof, and the Wall holds proof only.
    """
    certified = {e.get("source") for e in entries()}
    out = []
    for row in earned_skills.correlation_report():
        if row["id"] in certified:
            continue
        if row["served"] >= MIN_SERVES and row["rate"] is not None \
                and row["rate"] >= CERT_RATE:
            out.append(row)
    return out


def _certificate(row: dict, skill: dict) -> dict:
    tags = [t for t in skill.get("tags", []) if t]
    if "certificate" not in tags:
        tags.append("certificate")
    return {
        "id": f"cert-{row['id']}",
        "kind": "certificate",
        "title": f"Certificate of Mastery: {skill.get('title', row['id'])}",
        "tags": tags,
        "keywords": list(skill.get("keywords", [])),
        "text": ("Proven in service: laid beside %d task(s), %d sealed "
                 "complete (%.0f%%). This ground is earned, not guessed."
                 % (row["served"], row["complete"], 100 * (row["rate"] or 0))),
        "source": row["id"],
        "created": soul.now_ts(),
    }


def mint() -> list[dict]:
    """Mint a certificate for every eligible skill.  Idempotent: a skill is
    judged once, and a wall already holding its certificate refuses a second.

    Returns the certificates minted this pass.
    """
    skills = {s.get("id"): s for s in earned_skills.entries()}
    minted = []
    for row in eligible():
        cert = _certificate(row, skills.get(row["id"], {}))
        if _add(cert):
            minted.append(cert)
            try:
                from agent.spirit import beat
                beat("certified")
            except Exception:
                pass  # the wall stands whether or not the world turns
    return minted


def retrieve(task_text: str, *, k: int = SERVE_K_CERTS) -> list[dict]:
    """The certificate most worth laying beside this task, if any.

    Shares the Chronicle's scoring and its silence rule; unlike lessons and
    skills there is no reinforcement — a certificate is never judged twice,
    in either direction.
    """
    task_words = chronicle._words(task_text)
    scored = [(chronicle.score(cert, task_words), cert) for cert in entries()]
    hits = sorted([(s, c) for s, c in scored if s > 0],
                  key=lambda pair: pair[0], reverse=True)
    return [cert for _, cert in hits[:max(0, k)]]


def render(certs: Iterable[dict]) -> str:
    """The block laid into the prompt.  Empty string for no certificates."""
    certs = list(certs)
    if not certs:
        return ""
    lines = ["From your wall - ground you have truly earned:"]
    for cert in certs:
        lines.append(f"- {cert['title']}: {cert['text']}")
    return "\n".join(lines)


def _main(argv: Optional[list] = None) -> int:  # pragma: no cover - thin CLI
    import argparse
    parser = argparse.ArgumentParser(
        description="The Wall - certificates for skills proven in service")
    parser.add_argument("command", nargs="?", default="shelf",
                        choices=["shelf", "eligible", "mint"])
    args = parser.parse_args(argv)
    if args.command == "mint":
        minted = mint()
        if not minted:
            print("nothing newly earned - the wall does not certify "
                  "participation")
        for cert in minted:
            print(f"minted: {cert['id']}  {cert['title']}")
    elif args.command == "eligible":
        rows = eligible()
        if not rows:
            print("no skill yet meets the covenant (served >= %d, complete "
                  "rate >= %.0f%%)" % (MIN_SERVES, 100 * CERT_RATE))
        for row in rows:
            print("%s  served=%d complete=%d rate=%.0f%%" % (
                row["id"], row["served"], row["complete"],
                100 * (row["rate"] or 0)))
    else:
        certs = entries()
        if not certs:
            print("the wall is bare - proof comes with toil")
        for cert in certs:
            print(f"{cert['id']}  {cert['title']}")
            print(f"    {cert['text']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
