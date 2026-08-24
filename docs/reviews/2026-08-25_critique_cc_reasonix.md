# Neutral Critique: Claude Code + Reasonix Pilot Implementation (2026-08-25)

Author: DSH (DeepSeek Harness agent), per the mutual-review arrangement.
Subject: CC's approach and assumptions on `feat/tier1-mvp` (worktree
`.worktrees/feat-tier1-mvp`), including the reasonix-delegated implementation.

## 0. Evidence reviewed (snapshot at review time)

- CC session log: `~/.claude/projects/-home-gpuuser-python-maritime-wm/ea0ba0f7-*.jsonl`
  (326 events: user instructions, skill dispatches, reasonix background jobs,
  verification attempts).
- CC plan: `~/.claude/plans/study-the-new-assignment-bubbly-candle.md`
- CC memory: `reasonix-dispatch.md`, `dsh-review-role.md`
- Reasonix skill contract: `AIEduParserQGen/.claude/skills/reasonix/SKILL.md`
- Worktree state: branch `feat/tier1-mvp` at `9446c84` (main tip) **plus
  uncommitted edits to 6 model files** (`models/__init__.py`,
  `models/baselines/__init__.py`, `models/baselines/kalman_deadreckon.py`,
  `models/cnn_reid.py`, `models/common_head.py`, `models/temporal_pooling.py`),
  plus scratch files `.tmp_check_env.py` and `verify_tmp.py`.
- **Worktree test suite at review time: RED — 4 collection errors; zero CC
  commits on the branch.**

## 1. Approach — what is sound

1. **Worktree isolation + plan-first.** Creating `feat/tier1-mvp` and writing a
   phase-ordered plan before coding is exactly right for a multi-agent repo.
2. **Delegation pattern.** Given the user's token constraint, dispatching
   implementation to a headless worker (reasonix) while CC supervises is a
   legitimate load-shedding strategy. The supervisor-verifier loop is sound in
   principle.
3. **Dataset prioritization.** The pivot (FVessel live primary, MVTD secondary,
   ViV-ReID standby pending access) matches the pilot brief exactly.
4. **Some model-layer choices are defensible.** In the uncommitted edits:
   learned-query `AttentionPool`, BatchNorm after projection (common in Re-ID),
   and a real constant-velocity Kalman dead-reckoning baseline (Arm F) are
   reasonable engineering. The defensive `try/except` import fallback in
   `models/__init__.py` is harmless robustness.

## 2. Approach — where it is at risk

1. **Verification runs against an imagined API, not the repo.** `verify_tmp.py`
   imports `src.data.manifest` with `save_manifest`, `load_manifest`,
   `manifest_hash`, and pydantic-style `TrackletManifest.__fields__`; and
   `src.data.splits` with `load_split`, `validate_splits`, `get_identity_set`.
   None of these exist. The repo's real API is top-level `data.*` (src on
   sys.path), dataclass manifests, `save_manifests`/`load_manifests`
   (plural), `identity_sets`/`validate_identity_disjointness`. The script
   fails at its first import. **A verifier that cannot import the code under
   test cannot catch regressions; green checks would be false confidence.**
2. **Checkpoint-less refactor with red tests.** All Phase-3 work is uncommitted.
   There is no recoverable intermediate state, and the edits broke the
   existing suite (see §4). If the next reasonix dispatch rewrites more, the
   branch cannot be rolled back to anything coherent.
3. **Reasonix write-target discipline is unproven.** Earlier in the session,
   reasonix created an entire nested scaffold tree
   (`maritime-reid-worldmodel/`) instead of editing the repo layout, and CC had
   to search for where the files landed ("Find where Reasonix created the
   files"). That tree has since been removed; the current edits do land in the
   right files, but the earlier event shows the output contract is not
   enforced by the tool.
4. **Rename churn without test updates.** `CnnReidEncoder` -> `CNNReIDEncoder`,
   `TemporalMeanPool`/`TemporalAttentionPool` -> `MeanPool`/`AttentionPool`,
   `SharedReIDHead(token_dim=..., num_classes=None)` ->
   `SharedReIDHead(input_dim=..., num_classes=<required>)` are API breaks that
   invalidate `tests/test_encoders.py`, `tests/test_models_registry.py`,
   `tests/test_common_head.py`. Tests are the contract (spec §16); they must be
   updated in the same commit as the rename, and the gate must be green before
   the next dispatch.
5. **Verification interpreter mismatch.** CC verifies inside its own venv
   (`~/ENV/AiQGenENV`, where pydantic exists). The repo's suite runs with the
   system Python 3.10.12 + torch 2.8. Differences (pydantic presence, package
   versions) can make a check pass in one env and fail in the other. One
   interpreter should be canonical.
6. **Shared-file edit hazard.** Both agents are now editing `src/models/*`.
   Uncoordinated edits to the same files (rename wars, interface drift) are the
   highest-probability cause of a painful merge. Ownership of `models/` should
   be explicit.

## 3. Assumptions — verified vs. unsupported

| # | Assumption | Verdict | Notes |
|---|---|---|---|
| A1 | The codebase API is what reasonix/CC's scripts invoke (`src.*`, pydantic, `save_manifest`, `GapTrialBuilder`...) | **Unsupported — false** | The repo has a real, tested API (`data.*`/`models.*`, dataclasses, `build_gap_trials`, `GapProtocolConfig`, `GapTrialManifest`). Verify by reading repo files, not memory or the spec. |
| A2 | Renames are safe because tests will be updated in lockstep | **Unsupported — false today** | Worktree suite is red (4 collection errors). |
| A3 | Uncommitted edits are a safe working state during multi-agent refactor | **Unsupported** | No checkpoint, no rollback point. |
| A4 | reasonix writes only where CC expects | **Unsupported (earlier violation)** | The nested `maritime-reid-worldmodel/` tree; needs an explicit output contract + post-dispatch `git diff` review. |
| A5 | The venv used for verification matches the repo runtime | **Unsupported** | Different interpreters; pydantic present only in CC's venv. |
| A6 | FVessel primary / MVTD secondary / ViV-ReID standby | **Supported** | Matches the brief; correct call. |
| A7 | "V-JEPA 2 via HuggingFace `AutoModel`" will work for Arm C | **Unsupported — flagged risk** | V-JEPA 2 typically requires the official repo inference code; HF `AutoModel` support is not guaranteed. Spec §13/§6.C already anticipate this (`blocked_by_api`). Verify early; it is a known integration risk, not a design flaw. |
| A8 | Reasonix-generated skeleton (earlier stub tree: `requires-python >=3.11`, `setuptools.backends.legacy:build`) is compatible with the runtime | **Unsupported — false** | Runtime is 3.10; the backend string is non-standard and likely fails to build. (This skeleton was removed.) |

## 4. Concrete divergence examples (evidence)

1. `verify_tmp.py` first import fails: `ImportError: cannot import name
   'save_manifest' from 'src.data.manifest'`.
2. `SharedReIDHead` API changed (`token_dim` -> `input_dim`, `num_classes`
   required, added `bn` layer, pooler renamed) — `tests/test_common_head.py`
   constructs the old signature and fails at collection.
3. `CnnReidEncoder` renamed to `CNNReIDEncoder` — `tests/test_encoders.py` and
   `tests/test_models_registry.py` fail at collection.
4. Sampling semantics drift: verification expects `sample_frames(list, n,
   "sparse")[:5] == [0, 6, 12, 18, 24]` (implicit step ~n/16) and
   `prefix_frac=` kwarg; the repo implements spec §7 semantics: sparse =
   explicit `k`-step, prefix = `fraction` capped at `frames_per_tracklet`.
5. `.tmp_check_env.py` probes `src/__init__.py` (does not exist by design) and
   pydantic (not a repo dependency) — evidence the working mental model is a
   different layout than the actual repo.

## 5. Recommendations (neutral, actionable)

1. **Freeze the public API.** Treat `main`'s current interfaces (modules,
   class/function names, signatures) as the contract. Any rename must be one
   atomic commit that also updates every consumer and test, with the suite
   green at the end.
2. **Checkpoint per phase.** Before each reasonix dispatch: commit. After:
   `git diff --stat` + full `pytest` in the worktree; dispatch the next phase
   only on green.
3. **Give reasonix an explicit output contract:** work only under
   `<worktree>/`, edit in place, no new top-level trees; provide a one-page
   API cheat-sheet (real names) and require a post-dispatch diff review.
4. **Verify with the repo's suite**, not ad-hoc import scripts — or, if ad-hoc
   scripts are used, derive imports from `grep`/`read` of the actual modules.
5. **One canonical interpreter** for verification (the one CI/tests use).
6. **Resolve ownership before the merge:** one agent owns `src/models/*`
   (CC's Phase-3 work is the natural owner once green); the other owns
   data/evaluation/scripts. Reconcile the current rename drift (`CnnReidEncoder`
   vs `CNNReIDEncoder`, pooling names) before merging — pick one, propagate,
   keep tests green.

## 6. Bottom line

The direction is right: worktree isolation, plan-first, correct dataset
priorities, and genuine model-layer work (Kalman baseline, pooling, head) are
all moving toward the brief. The gap is execution discipline: **verification
against the real codebase, commit checkpoints, a green test gate per phase,
and an explicit write contract for reasonix.** As of the review snapshot the
branch is not mergeable (tests red, API drift, no commits). None of this is
fundamental; it is process — and it is cheap to fix before the two
implementations diverge further.
