# TASK-03 — Classical baseline: CVODE cost and stiffness at the frozen envelope

**Goal.** Extend `Surrogates/stiffness_benchmark.py` to the decision-1 mechanism
and envelope so that the denominator of every later speed claim is measured
under the exact conditions the surrogate will be tested on.
**Depends on.** TASK-02 (uses its config for envelope and mechanism).
**Effort.** 2 hours.

## Steps
1. Add the `ffcm1` and `gri30_cho` mechanisms with O2 oxidizer at p ∈ {100, 200, 300} bar,
   T0 ∈ {800, 1200, 2000} K, φ ∈ {0.5, 1.11, 2.0}; keep the existing air cases as regression.
2. Stiffness: worst, at-ignition, median, as the script already reports.
3. Cost: cold-start and warm-start CVODE per call at Δt ∈ {1e-8, 1e-7, 1e-6} s, at
   rtol/atol ∈ {(1e-6, 1e-12), (1e-8, 1e-15)}; 400 states per case; report µs/call.
4. Also time a batch of 10^4 cold-start calls through `ct.ReactorNet` in a tight
   loop to estimate Python overhead per call (subtract nothing; report both).

## Outputs
- `Surrogates/results/03-cvode-baseline.{json,md}`, `Surrogates/results/03-REPORT.md`.

## Done when
- The markdown table has one row per (mechanism, p, T0, φ, Δt, tolerance), and the
  worst-case fast eigenvalue at 300 bar is stated in s⁻¹ alongside the LES step it implies.

## Do not
- Do not change tolerances later to make the surrogate look better; these rows are frozen once written.
