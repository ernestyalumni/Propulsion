# TASK-04 — Tabulation incumbent (ISAT-lite)

**Goal.** A simple in-situ adaptive tabulation baseline so the surrogate is
compared with what production codes actually use, not only with direct integration.
**Depends on.** TASK-02.
**Effort.** 1 day.

## Method (keep it simple and honest)
- Work in the transformed space z = (T/1000, log(Y + 1e-20)) for the query key.
- Table entry: (z₀, R(z₀), A = ∂R/∂z₀ by central finite differences through CVODE, ε_tol).
- Query: nearest stored entry by Euclidean distance in z; if the linear estimate
  R(z₀) + A(z − z₀) is inside the entry's ellipsoid of accuracy (‖A(z − z₀)‖ ≤ ε_tol
  in the transformed output space), return it; else run cold CVODE, add an entry.
- ε_tol ∈ {1e-3, 1e-2}; table capacity 10^5 entries; report retrieve rate, add rate, accuracy, µs/query.

## Evaluation
- Run through the TASK-02 test split in trajectory order (so the table warms up as
  a CFD run would) and report: per-species log-relative error, T error, retrieve
  fraction, and mean µs/query including the CVODE calls on misses.

## Outputs
- `Surrogates/baselines/isat_lite.py`, `Surrogates/results/04-isat-lite.{json,md}`, `Surrogates/results/04-REPORT.md`.

## Done when
- The table rows exist for both ε_tol on the h2o2 and decision-1 sets, with retrieve fraction and cost, and the accuracy is reported on the held-out bands.
