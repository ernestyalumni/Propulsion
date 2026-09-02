# Physics generative surrogates: pre-story research handoff

**Status: discovery / hypothesis formation. Not implementation-ready.**

This directory hands the physics-generative-model investigation to another AI
session without prematurely treating it as an approved product feature. The
central question is whether an existing pretrained generative model—especially
a video or world model—can be adapted into a useful numerical surrogate for a
physics simulator.

The current answer is neither “yes” nor “no.” There are several different jobs
hidden inside that phrase, and they require different experiments:

1. generating physics code, derivations, solver configurations, and objectives;
2. replacing the stiff chemistry substep inside a classical reacting-flow
   solver;
3. predicting the future evolution of full spatial fields as a surrogate for a
   PDE solver;
4. generating visually plausible physical video for robotics or simulation
   workflows.

Only items 2 and 3 are numerical surrogates. Item 4 is not numerically accurate
merely because it looks physical.

## Reading order for the next session

0. [SESSION-2026-09-01-FINDINGS.md](SESSION-2026-09-01-FINDINGS.md) — **start
   here.** What was found, measured, decided, and built on 2026-09-01; where
   the AI is in each experiment; standing rules for agents; current state.
1. [tasks/README.md](tasks/README.md) — the task board: a launch preamble for
   any agent, the dependency graph, and one self-contained brief per task
   (TASK-00 … TASK-16).
2. [CANDIDATE-EXPERIMENTS-2026-09-01.md](CANDIDATE-EXPERIMENTS-2026-09-01.md)
   — the research memo: landscape audit, measurements, E1–E3 with
   preregistered go/no-go, decision recommendations (§6.1), state (§7).
3. [RESULTS-LEDGER.md](RESULTS-LEDGER.md) — every claim with its artifact.
4. [SESSION-HANDOFF.md](SESSION-HANDOFF.md) — the earlier session's complete
   technical and historical context (uses "Track A/B" for E1/E2).
5. [EVIDENCE-AND-OPEN-QUESTIONS.md](EVIDENCE-AND-OPEN-QUESTIONS.md) — claim
   ledger, corrections, primary sources; addendum F carries the 2026-09-01 corrections.
6. [GROK-BUILD-CONTINUATION.md](GROK-BUILD-CONTINUATION.md) — the prompt whose
   required output is now item 2; kept for provenance.
7. [../../PHYSICS-ML-BET.md](../../PHYSICS-ML-BET.md) — the earlier thesis
   document. Historical working material, not settled doctrine.
8. [../../../LaTeXandpdfs/SourceTermSurrogate.tex](../../../LaTeXandpdfs/SourceTermSurrogate.tex)
   — the mathematical case for the stiff-chemistry flow-map surrogate.

## Relationship to existing “stories”

Files 12–14 under `documents/stories/` were useful for extracting invariants and
preventing weak experiments, but they were written before the candidate model,
data representation, practical objective, and compute plan were settled. They
are now **provisional research contracts, not approved PDD user stories**. Do
not run `pdd intent apply` on them. Preserve them as source material until this
research graduates into one or more independently testable user stories.

The standing adversarial discipline in story 09 remains useful. Stories 10 and
11 are separate learning programs and do not need to block this discovery work.

## Graduation criteria

This work becomes ready for a user story only after all of the following are
decided:

- the job is named precisely: chemistry substep, full-field rollout, parameter
  inference, uncertainty ensemble, or something else;
- the target state, units, grid, boundary conditions, time step, and operating
  envelope are specified;
- the ground-truth solver and practical incumbent are named;
- the dataset exists or has a costed generation plan;
- the transferred model, scratch control, and specialist baselines are named;
- the success metric, failure margin, rollout horizon, and fallback behavior are
  fixed before training;
- local versus rented compute is estimated from a small measured run rather than
  from parameter count alone.

Until then, the correct deliverable is evidence and experimental design—not
training infrastructure or a large checkpoint download.
