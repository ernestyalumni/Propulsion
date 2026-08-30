# Chirikjian, *Stochastic Models, Information Theory, and Lie Groups* — reading guide

Two volumes, continuously numbered: **Volume 1** (*Classical Results and
Geometric Methods*, Birkhäuser 2009, 395 pp, chapters 1–9) and **Volume 2**
(*Analytic Methods and Modern Applications*, Springer 2012, 460 pp, chapters
10–22). Volume 2 opens at chapter 10, so a bare chapter number identifies the
volume unambiguously.

Corpus location (outside this repository, under the corpus root):

```
Public/books/EngineeringPhysics/Chirikjian-StochasticModelsInformationTheoryLieGroups-v1/
Public/books/EngineeringPhysics/Chirikjian-StochasticModelsInformationTheoryLieGroups-v2/
```

Each holds `INDEX.md` (grep-first table of contents), `toc.json`, and
`page_map.json`. **The printed-to-PDF page offset drifts** — Volume 1 runs from
+19 to +14, Volume 2 from +29 to +25, because unnumbered chapter title pages
shift the count. Never apply a single offset; read `page_map.json`, or the `PDF`
column in `INDEX.md`.

```bash
grep -i 'fokker' INDEX.md            # topic -> section + printed page + PDF page
grep -E '^\| 15\.' INDEX.md          # all of chapter 15
```

## What the two volumes actually are

A single argument, built in three movements.

1. **Volume 1, chapters 2–4** lay the classical ground: Gaussians and the heat
   equation, probability and information theory, then stochastic differential
   equations in Euclidean space — Itô and Stratonovich side by side (4.5, 4.6),
   ending with how SDEs and Fokker–Planck equations transform under a change of
   coordinates (4.8). That last section is the hinge for everything after it.
2. **Volume 1, chapters 5–8** replace Euclidean space with a manifold: curves
   and surfaces, differential forms, polytopes and manifolds (including fiber
   bundles and connections, 7.7), and then stochastic processes *on* manifolds
   (chapter 8) — the same Fokker–Planck story, now coordinate-free.
3. **Volume 2** replaces the manifold with a Lie group, which is where rigid-body
   motion lives, and then spends the second half on applications.

The through-line worth internalising is the SDE ladder:

| Setting | Where |
|---|---|
| SDEs in Rⁿ, Itô vs Stratonovich | **1**: 4.4–4.8 |
| SDEs on a manifold, Fokker–Planck, entropy | **1**: 8.1–8.4 |
| SDEs on a Lie group, unimodular case, CLT | **2**: 20.1–20.8 |

## Route: rigid-body motion and GNC

Read in this order; this is the spine of Volume 2.

- **2: 10** — matrix Lie groups and Lie algebras, change of basis (10.3), inner
  products on a Lie algebra (10.4), adjoint and Killing form (10.5), and the
  worked examples (10.6) where SO(3) and SE(3) appear as instances rather than
  as special cases to be memorised.
- **2: 11** — Lie derivatives, Taylor series on a group (11.3), the relationship
  between the Jacobian and Lie derivatives (11.4), sectional curvature (11.8).
- **2: 12** — integration, convolution, and Fourier analysis on groups. This is
  what makes a *probability density on SE(3)* a computable object.
- **2: 13** — variational calculus on Lie groups, deriving Euler–Lagrange
  (13.2) and the parameter-free treatment on matrix Lie groups (13.6). Euler–
  Poincaré lives here.
- **2: 20** — **the payoff for GNC.** Fokker–Planck on unimodular Lie groups
  (20.2), extracting Stratonovich SDEs back out (20.3), conditions for an Itô
  equation to stay on a matrix Lie group (20.4), measures of dispersion (20.6),
  and a central limit theorem for unimodular Lie groups (20.8).
- **2: 21** — the robotics chapter, and the reason 1: 7.7 matters: locomotion
  and perception as communication over a **principal fiber bundle**, with
  sensor fusion in mobile robotics (21.4).

**Why 20.2 is the one to read first if you only read one.** Propagating pose
uncertainty by keeping a 6×6 covariance in a local parameterisation is the
standard engineering move and it is wrong in a specific way: the covariance is
chart-dependent, so it drifts under composition and disagrees between
implementations. Chapter 20 gives the coordinate-free version — the density
evolves on the group itself, and 20.4 says exactly when an Itô equation stays on
the group instead of leaving it. This is the same concern as
`LaTeXandpdfs/SO3_SU2_Quaternions.tex` (naming the group before writing the
numerics) carried into the stochastic setting, and it is directly relevant to
story 8 in `documents/stories/`.

## Route: manufacturing and assembly

This is not a stretch — it is Chirikjian's own application, and it is the whole
of **2: 15**, *Parts Entropy and the Principal Kinematic Formula*.

- **15.2** problem formulation, **15.3–15.4** the principal kinematic formula
  (integral geometry: how often do two randomly placed bodies intersect).
- **15.11** kinematic formulas for **articulated** bodies.
- **15.12 Parts Entropy** and **15.13 Entropy of Loosely Connected Parts** —
  the information-theoretic cost of bringing parts from a disordered bin into
  a specified relative pose. This is a rigorous handle on bin-picking, feeder
  design, and assembly-sequence difficulty.
- **15.9–15.10** kinematic inequalities and bounds on integrals of powers of
  the Euler characteristic, if you want the sharp results rather than the
  formulation.

Supporting material: **1: 7.1** (convex polytopes) for the geometry of parts,
**1: 5.6** (Euler characteristic) for what the kinematic formula is counting.

## Route: materials, molecules, and continuum models

- **2: 14.4** equilibrium statistical mechanics of rigid-body molecules —
  a rigid body with a Boltzmann distribution over SE(3).
- **2: 14.5** conformational statistics of DNA; **2: 13.7** continuum models of
  DNA mechanics. Read these as the worked example of an elastic filament with
  thermal noise, not as biology: the machinery is Cosserat-rod-like and
  transfers to any slender structure under stochastic loading.
- **2: 14.6** ergodic theory — when a time average may be substituted for an
  ensemble average, which is the assumption every fatigue and life model makes
  silently.

## Route: financial markets

**Caveat, stated plainly: Chirikjian never discusses finance.** The transfer is
ours. What is genuinely portable is **2: 16**, *Multivariate Statistical
Analysis and Random Matrix Theory*:

- **16.5** integration and probability densities on **spaces of matrices**,
  **16.6–16.8** the Wishart distribution derived geometrically and applied.
  Sample covariance of a return series *is* Wishart-distributed under a Gaussian
  null, and the geometry of the positive-definite cone (16.7) is what makes
  shrinkage and conditioning arguments precise.
- **16.10** random matrix facts — the spectral results behind separating signal
  from noise in a large empirical covariance matrix.
- **16.3** resampling, **16.9** non-Gaussian multivariate statistics.

The honest summary: this chapter gives the *estimator geometry*, not a market
model. Volume 1 chapter 4 (Itô calculus, Ornstein–Uhlenbeck in 4.7) is the part
that looks like conventional quantitative finance, and it is standard there.

## Where this book set is stronger than the alternatives

- It derives Itô and Stratonovich **together** and then says precisely what
  changes under a coordinate transformation (1: 4.8) — most treatments pick one
  convention and leave the reader to discover the discrepancy.
- It builds fiber bundles (1: 7.7) *before* needing them, then actually uses
  them for a physical system (2: 21).
- Chapter 15 has no real substitute: integral geometry applied to assembly is a
  small literature and this is its textbook.

## Parse status

Both volumes are being parsed through the two-model OCR pipeline (Marker as
backbone, Nougat for equation cross-check, then reconciliation), because both
are equation-dense. The index in this guide is derived from the **embedded text
layer**, which is exact — both PDFs are born-digital LaTeX, not scans — so the
page maps here do not depend on the OCR run and will not change when it lands.
