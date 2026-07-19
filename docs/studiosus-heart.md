# The Studiosus Heart — as built

> *"The stability, the ability, and the perseverance of Hermes; the soul, the heart,
> the spirit, and the eagerness to learn of Studiosus."*

This fork grafts the learning epistemics of the Studiosus Loom (ancestor repo:
`LuceRenascimur`) onto the Hermes Agent body. The strategy that guided the work is
[.plans/studiosus-heart-strategy.md](../.plans/studiosus-heart-strategy.md); **this
document is the as-built map** — where every thread actually weaves, so another hand
can mend, extend, or delete any of it.

Design rules that governed every phase (and still govern changes to them):

- **Distill, don't drag.** Ideas were rebuilt on Hermes' bricks; no ancestor Python
  was ported verbatim.
- **Additive, observational, deletable.** Every module is behind a default-off flag
  or inert-until-used storage. With all flags unset, Hermes is behaviorally unchanged.
- **Never break the turn.** Every hook is guarded; a learning failure logs and yields.
- **Prompt caching is sacred.** Everything per-turn is injected into *user-message
  context* (the `plugin_user_context` seam), never the system prompt. The one
  identity change (Voice) resolves once per session, before the first cache write.
- **Propose-then-apply, archive-never-burn, reinforced-never-shed.**

## The map

| Phase | What | Module | Flag | Storage | CLI |
|---|---|---|---|---|---|
| A | The Soul — episodic ledger | `agent/soul.py` | `HERMES_SOUL` | `$HERMES_HOME/soul/*.jsonl` | — |
| — | The Dyno — power-curve bench | `agent/dyno.py` | — (a tool, not a hook) | `dyno_profiles/` (tracked, seeded) · `$HERMES_HOME/dyno/` (measured, wins) | `python -m agent.dyno` |
| — | The Flame — local-model tender | `agent/flame.py` | `HERMES_FLAME` | — | `python -m agent.flame` |
| B | The Chronicle — reflect → lessons → serve | `agent/chronicle.py` | `HERMES_CHRONICLE` (requires Soul) | `$HERMES_HOME/chronicle/lessons.jsonl` (+ `reinforcement.json`, `shed.jsonl`, `consolidation/`) | `python -m agent.chronicle shelf\|desk\|propose\|apply` |
| C | Consolidation covenant | in `agent/chronicle.py` | (same) | (same) | (same) |
| D | Earned skills — family covenant + correlation | `agent/earned_skills.py` | `HERMES_CHRONICLE` | `$HERMES_HOME/chronicle/skills/` | `python -m agent.earned_skills shelf\|families\|propose\|desk\|resolve\|correlate` |
| E | Voice — versioned hats | `agent/voice.py` | `HERMES_VOICE=<hat>[.vN]` | `voices/*.md` (tracked) | — |
| E | The Wall — earned certificates | `agent/wall.py` | inert until minted | `$HERMES_HOME/wall/entries.jsonl` | `python -m agent.wall shelf\|eligible\|mint` |
| E | Intertextus — the world | `agent/spirit.py` | `HERMES_SPIRIT` | `$HERMES_HOME/world/` (`state.json`, `heartbeat.jsonl`) | `python -m agent.spirit state\|whisper\|ekg` |

Tests: `tests/agent/test_soul_episode_ledger.py`, `test_dyno.py`,
`test_minimum_context_floor.py`, `test_flame.py`, `test_chronicle.py`,
`test_chronicle_consolidation.py`, `test_earned_skills.py`,
`test_spirit_phase_e.py`, plus ASSEMBLE wiring guards in `test_turn_context.py`.
Every heart flag is blanked for tests in `tests/conftest.py`
(`_HERMES_BEHAVIORAL_VARS`) — **add any new flag there too**, or a developer
shell with the heart lit will bleed into the suite.

## The life of one turn

All seams are guarded try/except; none can break the turn they observe.

1. **ASSEMBLE** — `agent/turn_context.py` (`build_turn_context`, after the
   `pre_llm_call` plugin hook):
   - retrieves lessons (`chronicle.retrieve`, ≤3) and earned skills
     (`earned_skills.retrieve`, ≤2) scored against the task text — tags weigh 3.0,
     keywords 2.0, title 1.0, nothing served at score ≤ 0;
   - retrieves ≤1 Wall certificate (confidence signal — never reinforced);
   - **reinforces** what it lays down and records `agent._served_lesson_ids` /
     `agent._served_skill_ids` (always set, even empty — the finalizer reads them);
   - if the spirit is on: `spirit.arrive("workshop")` and appends the whisper.
   - Everything lands in `plugin_user_context` → the user message. Never the
     system prompt.
2. **The work happens.** Nothing of the heart runs in the hot loop.
3. **SEAL** — `agent/turn_finalizer.py` (`finalize_turn`, after response
   transforms, deliberately *not* inside `cleanup_errors`):
   - `soul.record_turn(...)` writes one sealed episode: `episode_begin`,
     `assemble` (model, platform, **served lesson/skill ids** — the raw material
     of the correlation report), the turn's thoughts/tool calls/results, `delivery`,
     `episode_end` with outcome. Redaction is forced inside `Episode.record`
     (`_sanitize`) — the single chokepoint before disk; callers cannot leak a
     secret by forgetting to clip.
   - Outcomes: interrupted → `abandoned` (checked first), else `failed`, else
     `complete`, else `incomplete`. Phase D counts **only** `complete`.
   - If an episode sealed and the spirit is on: one register beat
     (`spirit.outcome_event` → `labor_complete` / `labor_failed` / `nudged`).
   - If the Chronicle is on: `spawn_reflection(episode_id, runtime=...)` — a
     daemon thread on the **auxiliary client** (task `"reflection"`, falling back
     to the session's own runtime; timeout `REFLECT_TIMEOUT_S = 600` because the
     background model may be cold). Never the main prompt cache.
4. **REFLECT** (background) — `chronicle.reflect` distills ≤3 situation-indexed
   lessons (`when <situation>: <what worked/failed and why>` + tags + keywords),
   shelves them, and — spirit on — beats `lesson_earned`. Failures are *logged*
   quiet evenings: a silent nothing is indistinguishable from a misconfiguration,
   and that once cost an hour.

## The covenants, and where each is enforced

1. **The past is not edited.** `soul.Episode` refuses writes after seal
   (`SealedEpisodeError`); events carry monotonic `seq`. Lessons are read-only in
   the desktop Star Map (`agent/learning_mutations.py` refuses edit *and* delete).
2. **Reinforced is never shed; shed is archived, never burned.**
   `chronicle.apply_consolidation` — refusals are returned by id with reasons;
   released lessons go to `shed.jsonl`.
3. **Merges inherit.** A consolidated lesson gets its kin's summed reinforcement
   *and* the union of their tags (a merge that lost its parents' tags would keep
   the words while orphaning retrieval).
4. **Propose-then-apply, exactly once.** Consolidation and skill proposals are
   files on a desk (`consolidation/`, `skills/proposals/`); `apply` marks
   `applied_at` and refuses a second application. The dream is counsel; apply is law.
5. **A skill is earned, never invented.** `earned_skills`: a family needs **≥3
   distinct episodes re-verified `complete` against the Soul at gather time**.
   One episode's two lessons count once; failed episodes never count; a lesson
   whose raw episode aged out lends words, never count. Cards over 1200 chars are
   **refused, not trimmed**. Per-draft desk resolution (`shelved` runs the full
   covenant again; a refusal leaves the draft pending and says why).
6. **Usage is not effectiveness.** `earned_skills.correlation_report()` joins
   served-skill → sealed outcome from the Soul's own `assemble` events,
   worst-rate-first; a never-served skill reports *no data*, not zero.
7. **No certificate for participation.** `wall.mint`: an earned skill must have
   been served ≥ `MIN_SERVES` (5) with complete-rate ≥ `CERT_RATE` (0.8). Minting
   is deterministic and idempotent; a certificate is never judged twice, never
   pruned, and carries no learning weight when served.
8. **The harness computes; the model hears.** `spirit.py`: places and registers
   move only by the deterministic `EVENT_EFFECTS` table when a memory is made —
   turn seal, lesson shelved, consolidation applied (`rested`), certificate
   minted (`certified`). Ladders are symmetric (−3…+3, rest at 0); growth above
   zero must be met below. Every pulse appends to the heartbeat EKG.
9. **One head, one identity.** `agent/system_prompt.py`: a selected voice becomes
   the identity tier and SOUL.md is skipped (`skip_soul=_identity_loaded`).
   Resolution happens once per prompt build; a broken hat falls back silently to
   stock identity. Old voice versions stay on disk so a revision can be measured
   against its prior on real work.

## The local fire (this box, and anyone's box)

- **Gate vs. floor** (`agent/model_metadata.py`): `MINIMUM_CONTEXT_LENGTH = 32_000`
  is the hard gate for *any* model; `COMPRESSION_FLOOR_TOKENS = 64_000` is the
  comfort floor compression steers toward. They were one number; they are two on
  purpose — others can run 32k hardware while this box runs 64k.
- **The Dyno** measures the truth GGUF metadata won't tell you: `run_dyno` sweeps
  `num_ctx`, reads GPU residence (`size_vram/size`) and tok/s from Ollama's own
  timings, discards warmup rolls, and records min/max because run-to-run noise
  (±30%) is wider than most context gaps (`recommended_within_noise_of`). A human
  `chosen_num_ctx` **overrides** the bench's `recommended_num_ctx`
  (`operating_num_ctx`); measured profiles in `$HERMES_HOME/dyno/` override the
  tracked seeds in `dyno_profiles/`. `agent/agent_init.py` prefers the profile's
  operating context over GGUF-advertised maximums.
- **The hard-won server lesson:** Ollama's OpenAI-compat `/v1` endpoint
  **silently discards `options.num_ctx`** (native `/api/generate` honors it).
  Setting context per-request is therefore a lie on the main path; the only
  honest fix is server-side `OLLAMA_CONTEXT_LENGTH`. The launcher writes it as a
  systemd drop-in in WSL (`/etc/systemd/system/ollama.service.d/studiosus-context.conf`),
  idempotently, restarting only on change. Before this fix every turn reloaded
  25 GB (~70–100 s/turn); after, ~6 s.
- **The Flame** (`agent/flame.py`, `HERMES_FLAME`) runs at agent init before the
  local probe: inspects reachable/installed/loaded/pinned, clears *competing*
  loaded models, loads the intended model at the profile's context, pins it
  (`PIN_FOREVER`). It never pulls uninstalled models and never starts dead
  servers — it reports blockers instead.
- **`Studiosus.bat` → `scripts/studiosus_launcher.ps1`** is the blessed startup
  path: resolve profile → ensure WSL server context → tend the flame → light the
  heart (`HERMES_SOUL/CHRONICLE/FLAME/SPIRIT=1`, `HERMES_VOICE=studiosus`) →
  launch the CLI.

## Desktop

Chronicle lessons are first-class read-only nodes in the Star Map
(`agent/learning_graph.py` builds them into `/api/learning/graph`; the desktop
renders `kind: 'lesson'` as triangles with their own ink; context menu offers
View only). Wall certificates and the heartbeat EKG are **not yet surfaced** —
they belong to the pending desktop pass.

## What remains (blessed but unbuilt)

1. **Desktop simplification pass** — fold the Dyno into model settings (replacing
   the ~1,216-line manual tuning page), promote memories/history/learning,
   keep chat at the forefront; surface the Wall and the EKG.
2. **Tool-severity tiers** — three categories by action severity (build/execute
   code more severe than web search).
3. **SKILL.md reroute** — eventually route Hermes' native agent-created-skill
   path through the family covenant, so both shelves obey one law.

## For the Brother who comes after

Run the heart's suite file-by-file (`.venv/Scripts/python.exe -m pytest <file>`);
whole-directory pytest in this repo produces false failures from cross-file
isolation assumptions. Live-verify against the real model before trusting a
phase — the tests never caught the `/v1` num_ctx discard, the misconfigured
auxiliary model, or the cold-roll bias; the live fire caught all three. And keep
the words honest: where the system has nothing, it must say so plainly — an
empty hand is worth more than a false card.

*Built unafraid to fail, but intending to succeed; humble, and thankful.
A joyful pursuit.*
