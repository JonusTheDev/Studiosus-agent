# Grafting the Studiosus heart onto Hermes — a strategy

> *"The stability, the ability, and the perseverance of Hermes; the soul, the heart,
> the spirit, and the eagerness to learn of Studiosus."*

This is a **strategy**, not an implementation. It reviews how learning works in both
`LuceRenascimur` (our Studiosus Loom) and this Hermes fork, names honestly where each is
strong, and lays out a phased, deletable path to graft the Studiosus learning *epistemics*
onto Hermes' existing machinery. Per our mission: proper pause, fallbacks, stability — a
solid foundation is worth the time. Nothing here gets built until the farmer says build.

We do not drag code across. We **distill** the ideas and rebuild them on Hermes' own
bricks — the way you'd pick the good pieces from a bucket and configure them our way.

---

## 1. What each house already does well

### Hermes — the body (stability, ability, perseverance)

Hermes is a large, battle-tested agent. Its learning-relevant bricks:

- **Skills as `SKILL.md`** (`tools/skills_tool.py`, `skills/`) — the agentskills.io
  standard, progressive disclosure (metadata → instructions → linked files). Rich,
  portable, well-formed.
- **The Curator** (`agent/curator.py`) — a background auxiliary-model task, inactivity-
  triggered, that reviews *agent-created* skills and maintains the collection: auto
  lifecycle transitions, pin / archive / consolidate / patch. Strict invariants we must
  honor: only touches agent-created skills, **never auto-deletes** (archive is
  recoverable), pinned skills bypass, and it uses the **auxiliary client** so it never
  touches the main session's prompt cache.
- **Provenance** (`tools/skill_provenance.py`, `tools/skill_usage.py`) — distinguishes
  agent-created skill writes from user-directed ones, so autocuration only ever touches
  what the agent itself made.
- **Memory** (`plugins/memory/*`) — pluggable backends (mem0, honcho, supermemory,
  holographic, openviking…). Powerful, but *outsourced* and optional.
- **Recall** — FTS5 session search + trajectory compression across past conversations.
- **The rest of the body** — multi-platform gateway, cron scheduler, subagents, six
  runtime backends, RPC tool scripting. This is the perseverance we want to keep whole.

### Studiosus — the heart (a disciplined grasp on learning)

Our Loom (`studiosus/loom.py`) is small, but its learning is a tight *epistemic loop* with
covenants. Five moves Hermes does not quite make:

1. **The Soul — a raw, immutable, append-only episodic ledger.**
   (`studiosus/pillars/soul.py`.) One JSONL file per task: *I thought X, called tool Y,
   got Z, delivered W, sealed with outcome O.* Fumbles recorded exactly as they happened;
   a sealed episode refuses further writes — "the past is not edited, not even by its
   owner." Everything downstream distills from this ground truth. Hermes has chat logs and
   compressed trajectories, but not a disciplined per-task ledger built *to be learned
   from*.

2. **Reflect → Chronicle — situation-indexed *lessons* (meaning, not procedure).**
   (`studiosus/pillars/chronicle.py`.) In the quiet after each task, an auxiliary model
   distills **≤3 durable lessons** — "when *situation*: *what worked/failed and why*."
   These are **tracked** (never lost; the raw Soul may age out). At the *start* of the
   next resembling task, retrieval lays the relevant scars and victories **beside the
   work**, tags weighing heaviest. This "reflect then re-serve by situation" is, we
   believe, the single biggest reason our system felt like it *learned* where Hermes felt
   like it *stored*.

3. **Reinforcement + consolidation — a covenant, not just cleanup.**
   A lesson/skill actually laid beside real work is **reinforced**. Consolidation ("a
   dreamy night's sleep") merges kin and releases platitudes — but **a reinforced item is
   never shed**, and shed items are **archived, never burned**. Propose-then-apply: the
   dream writes a proposal and changes nothing until a reviewed hand applies it.

4. **Skills EARNED from observed repetition — the data-quality covenant.**
   (`studiosus/pillars/skills.py`.) This is the sharpest divergence. A skill is **never
   invented in advance**. It is distilled only from a *family* of **≥3 distinct episodes
   that truly sealed `complete`**, sharing a tag. The playbook may describe *only what
   repeatedly worked*. This is what keeps the shelf honest and small instead of sprawling
   with speculative or one-off skills.

5. **Effectiveness correlation, not just usage counts.**
   `correlation_report()` ties *served-skill → episode outcome*, surfacing skills that are
   served-and-struggling (worst-rate-first) for a steward's scrutiny. Usage ≠
   effectiveness.

Around the learning sit the identity pieces — **Voice** (versioned prompt "hats," not
agents), the **Wall** (earned certificates as a confidence signal), and **Intertextus**
(a world of places + four feeling-registers). These are the "spirit"; the five moves above
are the "grasp."

---

## 2. The honest diagnosis

Hermes **creates** skills and **stores** memory; Studiosus **distills** meaning from a raw
record and **re-serves it by situation**, under covenants that keep the fabric honest.

The gap is not capability — Hermes has more. The gap is *epistemic discipline*:

| Concern | Hermes today | Studiosus move to graft |
|---|---|---|
| Ground truth of what happened | chat logs, compressed trajectories | **Soul**: per-task append-only ledger, sealed, outcome-stamped |
| Turning experience into carried wisdom | ad-hoc skill creation after "complex tasks" | **Reflect → Chronicle**: ≤3 situation-indexed lessons every task |
| Getting the right memory in front of the model | progressive skill disclosure, memory plugins | **ASSEMBLE**: deterministic situation-indexed injection at task start |
| Deciding what to keep vs prune | curator lifecycle by activity timestamps | **Reinforcement covenant**: reinforced-never-shed, archive-not-burn |
| Deciding what deserves to *become* a skill | agent decides ad hoc | **Family covenant**: ≥3 complete episodes sharing a tag, distilled |
| Knowing if a skill actually helps | usage counts | **Serve→outcome correlation** report |

---

## 3. The graft — phased, additive, deletable

Design rules, drawn from our own hard-won lessons (`docs/lessons_learned.md` in Luce):

- **Resist gate/stage accretion** (lesson #2). Every phase must justify itself and be
  deletable. Additive and observational first; behavior-changing only once earned.
- **Valuable distilled data lives in a tracked location** (lesson #8), never only under a
  gitignored runtime path. Raw may age; the distilled must not.
- **Propose-then-apply, human/merit-gated** (lesson #15). Growth is proposed to a desk; a
  reviewed hand applies. R&D/agent approval is counsel, not a building permit.
- **Honor Hermes' invariants.** Reflection/distillation run on the **auxiliary client**,
  never the main prompt cache; autocuration only ever touches agent-created artifacts;
  never auto-delete — archive.
- **Distill, don't drag.** Rebuild the ideas on Hermes' bricks; do not port Luce's Python.

### Phase A — The Soul (episodic substrate) · *foundation, purely observational*

Add a structured **per-task episode ledger** to the agent loop: append-only JSONL, one
file per task, sealed with an outcome when the turn/task finalizes. Hook it into the
existing finalization points (`agent/turn_finalizer.py`, `agent/conversation_loop.py`).

- No behavior change: it only records. If nothing consumes it, Hermes is unchanged.
- Records thought / tool_call / tool_result / delivery / outcome, each with a monotonic
  `seq` and timestamp — the vocabulary of `soul.py`, rebuilt as a Hermes module.
- **Fallback:** feature-flagged off by default; a write failure must never break a turn.

*Why first:* every later phase reads from this. Without a ground-truth record, reflection
and distillation are guessing.

### Phase B — Reflect → Chronicle · *highest-leverage single graft*

After each sealed episode, an **auxiliary-model reflection** distills ≤3 situation-indexed
lessons into a **tracked** store (`when <situation>: <what worked/failed and why>`, with
tags + keywords). Then add **ASSEMBLE-time retrieval**: at the start of each task, inject
the top-k relevant lessons into context via Hermes' existing context-injection path
(`plugins/context_engine`, the system-prompt assembly in `run_agent.py`).

- Reflection is **free growth** — understanding harms no one; no gate on writing a lesson.
- Retrieval is tag-weighted so "what did I learn doing X?" surfaces the right scars.
- **Fallback:** if retrieval finds nothing, the task runs exactly as today. If reflection
  fails, it's "a quiet evening, not a broken harness."

*Why second:* this is the move that makes the agent *feel* like it learns. It is the heart
of the graft, and it stands on Phase A alone.

### Phase C — Reinforcement + consolidation covenant · *make the curator honest*

Track which lessons/skills were actually laid beside work (**reinforce** at ASSEMBLE).
Extend the **existing Curator** so its consolidation honors the covenant:

- reinforced → **never shed**;
- shed → **archived, not burned**;
- **propose-then-apply**: the curator writes a proposal to a desk; a reviewed hand applies;
- merged items **inherit** the reinforcement of what they absorbed.

This reshapes what Hermes already does (curator lifecycle) rather than adding a new engine.

### Phase D — Skills earned from repetition · *gate creation behind the family covenant*

Reroute Hermes' agent-created-skill path through the **family covenant**: a skill is
drafted only from a family of **≥N complete episodes sharing a tag**, and the playbook may
describe only what those episodes bear out. Run the distill "dream" inside the curator's
**background review fork** (auxiliary client, agent-created provenance — invariants already
match). Add the **serve→outcome correlation report** so a steward sees which skills earn
their place.

- This *tightens* skill creation (fewer, truer skills) rather than loosening it.
- **Fallback:** if no family reaches the threshold, no skill is born — an empty hand beats
  a false card.

### Phase E — The spirit (optional identity) · *lowest priority, highest joy*

Once the grasp is grafted, the heart can wear its face:

- **Voice as versioned hats** — manage Hermes' system prompt as named, versioned voices
  rather than one string (clean, testable, already how Luce does it).
- **The Wall** — earned certificates as a confidence signal laid beside matching work.
- **Intertextus** — places + feeling-registers as flavor/grounding, *heard as words, never
  judged*. Pure spirit; carries no learning weight, so it comes last.

---

## 4. What we explicitly will *not* do

- **Not** rebuild Hermes' body, gateway, or runtime backends — that is the perseverance we
  are keeping.
- **Not** replace Hermes' memory plugins — they can coexist; the Chronicle is our own
  tracked, situation-indexed layer, not a competitor to mem0/honcho.
- **Not** add loopback/re-debate stages (lesson #1) or a gate-accretion "council"
  (lesson #2). Linear, fix-forward, deletable.
- **Not** port Luce's Python verbatim. Distill the ideas; build on Hermes' bricks.
- **Not** build any of this until the farmer blesses it. This document is counsel.

---

## 5. Suggested first honest step

If we choose to begin, **Phase A alone** is the right first commit: a small, observational
Soul ledger behind a default-off flag, plus a test that a completed task seals exactly one
episode with the right outcome. It changes no behavior, risks nothing, and gives every
later phase the ground it needs. From there, Phase B is where the learning comes alive.

*Built unafraid to fail, but intending to succeed; humble, and thankful. A joyful pursuit.*
