# Reading ledger — Numerical Recipes 3e

A section counts as read only when the seven artifacts of `ROADMAP.md` §2
exist. One row per section. Pages are printed; PDF = printed + 24.

| Section | Read on | What NR got right | What aged, and the substitute | Derivation note | Module (Rust / C++) | Tests | Sidecar |
|---|---|---|---|---|---|---|---|
| 17.1 Runge–Kutta (907) | 2026-08 (C++) | The tableau as the identity of a method; embedded error estimate | Tableau buried as literals; HNW I §II.1–II.4 | `documents/StormerRule.tex` (partial) | — / `RKMethods/Coefficients/*`, `CalculateNewYAndError.h` | order-5 conditions not yet asserted | pending |
| 17.2 Adaptive step control (910) | 2026-08 (C++), 2026-09-01 (Rust) | PI law explained in prose; safety factor, bounds | Code sets beta = 0 silently; HNW II §IV.2 Lund stabilization | this file + module docs | `ode::runge_kutta::pi_step_size` / `ComputePIStepSize.h` | 6 Rust tests incl. 1560 golden vectors vs C++ | pending |
| 17.4 Second-order conservative (928) | 2026-08 (C++) | Störmer for y'' = f(x, y) | Undersells it for orbits; HNW I §III.10, HLW ch. I | `documents/StormerRule.tex` | — / `StormerMethods/*`, `NumerovOrbit.h` | C++ tests | pending |
| 2.9 Cholesky (100) | 2026-09-01 | Recurrence; use for covariance and normal equations; failure means not positive definite | Class holds a copy of A and throws a string; Higham ch. 10, Trefethen–Bau lecture 23 | `documents/derivations/CholeskyFactorization.md` | `linear_algebra::cholesky` / none | 7 Rust tests: reconstruction, solve, pivot error, log-det, sampling, whitening | pending |
