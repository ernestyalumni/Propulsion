# TASK-08 — Gate, fallback, a posteriori rollouts, and the go/no-go table

**Goal.** Decide E1 against the preregistered criteria in
`CANDIDATE-EXPERIMENTS-2026-09-01.md` §4 E1.
**Depends on.** TASK-06, TASK-07 (and TASK-03/04/05 rows for the table).
**Effort.** 2 days.

## Gates
1. Mahalanobis: fit mean and covariance of the training inputs in transformed
   space; threshold at the 99.9th percentile of training distances.
2. Ensemble disagreement: the three TASK-07 seeds; defer if max pairwise ψ-space
   distance exceeds a threshold set on validation.
3. Admissibility: any negative Y after projection, any Newton failure, any T' outside envelope → defer.
Deferral = run cold CVODE. Report deferral fraction d and compare with
d_break = (1 − g − r)/(1 − r), where g = gate cost / CVODE cost and r = surrogate cost / CVODE cost.

## A posteriori tests
1. 0-D rollouts: from held-out initial conditions, step the surrogate for 10^3–10^5
   steps at Δt = 1e-7 and 1e-6 s; compare with CVODE on ignition delay, T(t),
   major and radical species, enthalpy drift, ‖EΔY‖∞ accumulated; record the first
   time any quantity exceeds tolerance.
2. 1-D flames (TASK-06): replace the chemistry substep; flame speed and profiles vs
   the CVODE-substep run; chemistry share f; end-to-end speedup on the 1-D solver.
3. Cost: batched surrogate per cell on the 3060 vs cold CVODE from TASK-03; include
   projection, Newton, and host–device transfers.

## The table (fill every cell; blanks are failures)
| Criterion | Threshold | h2o2 | decision-1 mechanism |
|---|---|---|---|
| ignition delay error on held-out bands | ≤ 2% | | |
| T error | ≤ 1% | | |
| fallback fraction d | < 5% and < d_break | | |
| per-cell cost vs cold CVODE at production tolerance | ≥ 50x cheaper | | |
| 1-D flame speed error | ≤ 3% | | |
| Stiff-PINN and ISAT-lite on same axes | present | | |

## Outputs
- `Surrogates/chem/gate.py`, `Surrogates/chem/rollout.py`, `Surrogates/results/08-gonogo.{json,md}`, `Surrogates/results/08-REPORT.md`.

## Done when
- The table is complete with a one-word verdict per mechanism (GO / NO-GO) and the reason for any NO-GO. A NO-GO is a valid, complete result.
