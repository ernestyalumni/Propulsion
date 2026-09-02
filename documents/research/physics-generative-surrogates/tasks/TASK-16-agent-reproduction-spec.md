# TASK-16 — Agent-reproduction contract for E3

**Goal.** A written contract for the experiment in which an LLM agent, given
only story 12 and the E1 benchmark contract, must reproduce the TASK-02 to
TASK-08 pipeline and match its numbers.
**Depends on.** TASK-08 complete (the reference numbers must exist).
**Effort.** 1 day. No GPU.

## Write `Surrogates/agent/CONTRACT.md` with
1. Inputs the agent receives: `documents/stories/12-*.md`, the frozen benchmark
   contract (envelope, mechanism, tolerances, splits, metrics, margins), and the
   Surrogates venv. Nothing else from `Surrogates/chem/`.
2. Outputs it must produce: the same artifacts as TASK-02, 03, 07, 08 under
   `Surrogates/agent/run-<date>/`.
3. Pass criteria: every TASK-08 table cell within stated tolerances of the
   reference (e.g. ignition-delay error within 0.5 percentage points, cost within 20%),
   the validator passing, the provenance recorded, no hand edits by a human.
4. What counts as failure: silent skipping of a baseline, changing a tolerance,
   touching the held-out bands, or unreproducible numbers on rerun.
5. Which harnesses to try first: Claude Code with this repository; OpenClaw with a
   local open-weight model; note DeepFlame 2.0's DFODE-kit Trainer agent as prior art.

## Outputs
- `Surrogates/agent/CONTRACT.md`, `Surrogates/results/16-REPORT.md`.

## Done when
- The contract is complete enough that a second person could grade an agent run without asking questions.
