# Candidate experiments — 2026-09-01

**Prepared by:** Claude Code session, 2026-09-01
**Inputs:** every file named in `GROK-BUILD-CONTINUATION.md`, a fresh primary-source
landscape check dated 2026-09-01, and one measurement run
(`Surrogates/stiffness_benchmark.py`, results in `Surrogates/results/`).
**Nothing downloaded:** no model weights, no datasets, no cloud. One uv venv was
created at `Surrogates/.venv` with Cantera 3.2.0, NumPy 2.5.2, SciPy 1.18.1.

## 0. Decision in one paragraph

"Beat PINN and FNO" is the wrong finish line, and the 2026 literature has moved
past it. Beating a learned baseline on a held-out MSE is the ML-versus-ML claim
that McGreivy and Hakim showed is weakly informative. The scoreboard that a
propulsion engineer cares about is wall time at matched physical error against
the tuned classical solver, end to end, including break-even count. On that
scoreboard the only track this repository can win locally this quarter is the
stiff-chemistry flow-map surrogate (E1 below): truth is cheap, the incumbent is
measured (this memo), the physics argument is already written, and PINN and
FNO-class methods can be run as mandatory baselines on the same axes. The
text-pretrained LLM enters as the engineer that builds and validates that
surrogate, not as the kernel. The video-pretraining question (E2) is real,
narrower than the handoff thought, and should run second on the 3060 with
physics-pretrained open weights that did not exist in the ledger. Cosmos3-Edge
is demoted from candidate to "not a numerical-field candidate until a reason
appears."

## 1. What changed since the handoff (primary sources, checked 2026-09-01)

### 1.1 Story 13's question has been partly answered by others

Sotoudeh, Mukhopadhyay, Ohana, McCabe, Lawrence, Ho, Cranmer, *On the Value of
Tokeniser Pretraining in Physics Foundation Models*, arXiv:2603.05598 (ICLR 2026
AI & PDE workshop). Simplified MAGVIT-2 causal-conv tokeniser without VQ or
adversarial losses. Four Well datasets. VRMSE after 10,500 rollout-training
steps on Euler multiquadrants:

| Tokeniser initialization | VRMSE | vs scratch |
|---|---|---|
| scratch | 0.439 | — |
| in-domain physics pretraining, trainable | 0.158 | −64% |
| in-domain, mostly frozen | 0.162 | −63% |
| out-of-domain physics (RB + active matter + shear) | 0.355 | −19% |
| natural video | **not tested** | — |

Consequence: the interesting open question is no longer "does pretraining help
a tokeniser" (yes, strongly when in-domain). It is "does natural-video
pretraining beat cross-domain physics pretraining, or even scratch, at matched
architecture and budget." PhysiX says video init helped; Sotoudeh et al. did not
test it. That is E2, and it is smaller than story 13 as written.

### 1.2 Physics-pretrained open weights now exist that fit the 3060

These were absent from the candidate matrix in `EVIDENCE-AND-OPEN-QUESTIONS.md`
section C and they dominate Cosmos3-Edge as numerical-field candidates:

| Model | Size | License | Input | Notes |
|---|---|---|---|---|
| GPhyT (FloWsnr/General-Physics-Transformer; HF `flwi/Physics-Foundation-Model`) | S 9.2M, M 112M, L 385M | MIT | 4 frames, 256×128, multi-field | neural differentiator + Forward Euler; trained 1M steps on 4×H100/A100; headline in v4 abstract is "more than 7x" lower **next-step NMSE vs DPOT**, not a solver comparison |
| PDE-Transformer (thuerey-group, HF `thuerey-group/pde-transformer`) | 33M–701M (6 variants) | MIT | regular grids, 4×4 patches, PDE-parameter conditioning | arXiv:2505.24717 |
| Walrus (PolymathicAI, HF `polymathic-ai/walrus`) | 1.3B | MIT | short snapshot history, 2D as thin 3D, vector/tensor-aware | arXiv:2511.15684; 19 Well scenarios; transferred to lab Rayleigh–Taylor with ≤3 DNS realizations (arXiv:2606.01470) |
| Cosmos3-Edge (nvidia) | 4B, released 2026-07-20, updated 2026-08-25 | OpenMDW 1.1 | RGB image/video, text, action; diffusion tower for continuous modalities | BF16 only tested; no field interface; no verified fine-tune recipe for our use |
| PhysiX (arshka/PhysiX) | 4.5B | — | discrete Cosmos-1.0 tokens | **no checkpoints released** (rechecked); depends on archived `Cosmos-1.0-Autoregressive-4B`, `Cosmos-1.0-Tokenizer-DV8x16x16` / `CV8x8x8`; recipe uses `torchrun --nproc_per_node 8` |

Cosmos3-Edge is therefore no longer the first current-weight probe. Its VAE
remains a possible *natural-video* arm in E2, but a physics-pretrained model at
one tenth to one four-hundredth the size is the stronger transfer candidate.

### 1.3 The 2026 evaluation literature converged on one scoreboard

- **Breakeven complexity** (arXiv:2605.15399, May 2026): counts forward solves
  before a learned solver is cost-effective against an *error-matched*
  classical solver, including data generation, training, and tuning. Finding:
  neural solvers pay off sooner as problems get harder (cost, dimension,
  rollout, Reynolds number). APEBench PDEs plus a PyFR multi-obstacle benchmark.
- **No Free Lunch in Flow Surrogates** (arXiv:2607.23667, Jul 2026): eight
  architectures, two regimes (3D CMP slurry film; 2D Kármán street); surrogates
  are 10^3–10^4 faster than the FE solver and break even from the first or third
  query, but **no architecture won both regimes**.
- **When a neural surrogate cannot accelerate a solver** (Thümmler and Kuroda,
  arXiv:2608.23075, Aug 2026). A controlled negative result that is the closest
  published analogue to story 12: a 1.8M-parameter residual MLP replaced the
  implicit Newton solve for neutrino–matter coupling in a GR
  radiation-hydrodynamics code. The block was 5.8x cheaper per call
  (5.9e-5 s vs 3.4e-4 s) but occupied only 16.9% of critical-rank wall time, so
  Amdahl capped acceleration at ~1.2x; a Mahalanobis gate deferred 96.8–99.7%
  of cells because visited states sat 73x off the data manifold, making the
  gated loop a 0.94–0.96x slowdown; density bias drifted linearly to −19.9%
  over 6000 steps; and offline error did not rank surrogates by survival
  (rho +0.73 confounded, −0.04 controlled). Break-even deferral fraction
  d_break = (1 − g − r)/(1 − r), with g the gate cost ratio and r the surrogate
  cost ratio; gating is unviable when g > 1 − r.
- **Predictivity and Utility of Neural Surrogates of Multiscale PDEs**
  (arXiv:2604.20061): spectral bias, favourable benchmarks live on
  low-dimensional solution manifolds, coarse-graining loss is irreversible,
  weather reanalysis is a sweet spot that will not generalize to truly chaotic
  multiscale problems; recommends hybrids and reporting standards.

Every one of these is consistent with the standing rules in section E of the
ledger. Two of them (Thümmler–Kuroda; No Free Lunch) supply numbers to
preregister against.

### 1.4 The text-LLM half, updated

- Liu et al., *LLMs learn governing principles of dynamical systems, revealing
  an in-context neural scaling law*, arXiv:2402.00795: LLaMA-2 predicts
  dynamical-system time series in context without fine-tuning and improves with
  context length. This is the strongest text→numerics evidence and it is
  zero-shot in-context, slow, and not compared with a small specialist at
  matched compute. It does not support a fine-tuned text LLM as a PDE kernel.
- LLM-ODE (arXiv:2603.20910) and LLM-ACES (arXiv:2606.25039): LLMs guiding
  symbolic discovery of ODEs. Authoring, not simulation.
- **DeepFlame 2.0** (deepmodeling blog, 2026-01-28) ships a "DFODE-kit Trainer
  agent" that sets up operating conditions, generates data, trains, and
  validates a combustion-chemistry DNN from natural-language interaction. That
  is exactly the defensible form of the thesis: the LLM is the engineer, a
  small supervised net is the kernel, the classical solver is the baseline.
  It is also prior art for E3.

### 1.5 Ledger corrections

- **A.5 GPhyT number.** The v4 abstract fetched 2026-09-01 reads "outperforming
  specialized architectures by more than 7x." The body identifies the metric as
  next-step NMSE and the denominator as DPOT (next best). The "29x" recorded
  earlier was not found in v4; do not cite it without a version and table.
- **Section C candidate matrix** lacks GPhyT, PDE-Transformer, Walrus. Add them
  above Cosmos3-Edge for Track B.
- **"PhysiX provides a ready-to-run checkpoint": still not found** as of
  2026-09-01 on the GitHub README.
- **DFODE-Kit facts** (repo README): truth labels via Cantera/CVODE; HDF5
  datasets; augmentation preset `random-local-combustion-v1`; GPL-3.0; full
  workflow needs OpenFOAM + DeepFlame + Conda. Isolated Cantera labeling does
  not need OpenFOAM.

## 2. Measured today: the number story 09 demanded

`Surrogates/stiffness_benchmark.py` (Cantera 3.2.0 CVODE, constant-pressure
homogeneous ignition, fuel with air `O2:1, N2:3.76`; central-difference Jacobian
of S(φ) in (T, Y) coordinates; the n_e+1 eigenvalues nearest zero are the
element and enthalpy invariants and are excluded from the slow-mode
denominator). Raw output: `Surrogates/results/stiffness_benchmark.json`.

### 2.1 Stiffness ratio ς = max|Re λ| / min_active|Re λ|

| Mechanism | T0 [K] | p [atm] | φ | τ_ign [s] | ς worst (where) | ς at ignition (fast / slow, 1/s) | ς median |
|---|---|---|---|---|---|---|---|
| h2o2 (10 sp) | 1000 | 1 | 1.0 | 3.12e-4 | 3.6e8 (t/τ=0.01) | 2.5e3 (6.7e7 / 2.7e4) | 2.1e8 |
| h2o2 | 1200 | 1 | 1.0 | 4.53e-5 | 8.6e6 (0.13) | 1.4e3 (7.5e7 / 5.5e4) | 8.0e6 |
| h2o2 | 1500 | 1 | 1.0 | 1.30e-5 | 2.7e5 (0.27) | 8.7e2 (7.9e7 / 9.0e4) | 2.3e5 |
| h2o2 | 1200 | 1 | 0.5 | 4.75e-5 | 7.7e6 (0.13) | 5.3e2 (4.5e7 / 8.4e4) | 7.1e6 |
| h2o2 | 1200 | 1 | 2.0 | 5.09e-5 | 1.1e7 (0.10) | 3.3e3 (8.9e7 / 2.7e4) | 1.1e7 |
| h2o2 | 1200 | 10 | 1.0 | 6.38e-5 | 8.6e6 (0.00) | 1.0e3 (5.7e8 / 5.7e5) | 6.9e6 |
| gri30 (53 sp) | 1400 | 1 | 1.0 | 3.44e-3 | 7.6e12 (0.00) | 2.0e5 (5.0e8 / 2.5e3) | 2.2e10 |
| gri30 | 1400 | 10 | 1.0 | 4.99e-4 | 1.5e12 (0.01) | 1.0e5 (2.1e9 / 2.0e4) | 1.3e10 |

Reading. The ς ~ 10^8 (H2) and ~10^12 (hydrocarbon) figures in
`SourceTermSurrogate.tex` are reproduced, but they are induction-period values
where the slow modes are nearly frozen (|Re λ| of 10^-2 to 1 s^-1). At ignition,
where heat release lives, ς is 10^3 for H2 and 10^5 for GRI-3.0. Both readings
matter and must be reported together: the residual-conditioning argument
(κ ≳ ς²) is hopeless either way for first-order optimisation, and the flow-map
argument gets *stronger* exactly where ς is largest. The number to quote in
story 09 is therefore a range with the trajectory position attached, not a
single figure. Fast eigenvalue at ignition: 7e7 to 9e7 s^-1 for H2 at 1 atm,
5.7e8 at 10 atm, 5e8 to 2e9 for methane — the fast time scale is 1–20 ns.

### 2.2 The incumbent's cost (single core, Cantera Python API, this machine)

| Mechanism | Δt [s] | rtol / atol | warm-start µs/call | **cold-start µs/call** |
|---|---|---|---|---|
| h2o2 | 1e-6 | 1e-6 / 1e-12 | 21.4 | **127.6** |
| h2o2 | 1e-6 | 1e-8 / 1e-15 | 39.8 | 185.0 |
| h2o2 | 1e-7 | 1e-6 / 1e-12 | 3.0 | 92.7 |
| h2o2 | 1e-7 | 1e-8 / 1e-15 | 4.7 | 125.6 |
| gri30 | 1e-6 | 1e-6 / 1e-12 | 6.2 | **1335** |
| gri30 | 1e-6 | 1e-8 / 1e-15 | 9.8 | 1912 |
| gri30 | 1e-7 | 1e-6 / 1e-12 | 2.1 | 1058 |
| gri30 | 1e-7 | 1e-8 / 1e-15 | 2.3 | 1299 |

Warm start keeps CVODE's step history across calls along one trajectory; cold
start resets the reactor state and reinitializes the integrator before every
call, which is what an operator-split CFD solver does in every cell at every
step. **Cold start is the denominator a surrogate replaces.** Python overhead is
included; a C++ caller will be faster by a small factor, which must be measured
before any speed claim. Cantera's default tolerances are tighter than either
row; the "production" row (1e-6/1e-12) is the honest baseline unless the
application needs more.

### 2.3 What this buys the plan

- Truth generation for E1 is trivially cheap: 10^6 cold-start H2 labels cost
  about 130 s on one core; 10^6 GRI-3.0 labels about 22 min. The dataset
  criterion in the README ("exists or has a costed generation plan") is met.
- A batched MLP on the RTX 3060 at 10^5 cells per batch should land at 0.1–1 µs
  per cell, i.e. 10^2–10^3 per-call versus cold CVODE for H2. That reproduces
  DFODE-Kit's O(10^2) isolated figure *before* any Amdahl accounting.
- Amdahl: with chemistry share f of wall time and per-call speedup S,
  end-to-end speedup is 1/((1−f) + f/S). f = 0.8 gives ≤ 5x; f = 0.95 gives
  ≤ 20x. The chemistry share of the host solver must be profiled before an
  end-to-end number is promised. Thümmler–Kuroda's 16.9% is the cautionary
  bound.

## 3. Reframing "beat PINN and FNO"

There are four scoreboards. Only one matters for propulsion, and it is not the
one the phrase names.

| Scoreboard | What "beat" means | Who already won it |
|---|---|---|
| ML vs ML on next-step error | lower NMSE/VRMSE than FNO/PINN on a fixed dataset | Transformers with in-domain physics pretraining (GPhyT, Walrus, PDE-Transformer). Architecture-reuse half of the bet is settled in the literature; do not re-prove it. |
| Stiff kinetics, residual vs flow map | train at all | Flow-map supervised surrogates (DFODE-Kit and predecessors). Stiff-PINN (Ji et al., arXiv:2011.04520, DENG-MIT/Stiff-PINN) only works after QSSA removes the stiffness. |
| Wall time at matched error vs tuned solver, end to end | fewer seconds per acceptable answer, including break-even | Rare, regime-dependent (Breakeven; No Free Lunch; Thümmler–Kuroda negative). **This is the one to win.** |
| Authoring/orchestration | an LLM produces a validated surrogate pipeline from intent | DeepFlame 2.0 Trainer agent (Jan 2026). Where a text LLM genuinely beats hand-built PINN/FNO workflows. |

So: run PINN (Stiff-PINN with the authors' code) and FNO/TFNO as *mandatory
baselines on the same axes*, and define winning as beating the classical solver
at matched error on a named workload with the break-even count reported.

## 4. Candidate experiments

### E1 — Stiff-chemistry flow-map surrogate, adversarially benchmarked (engineering track)

- **Value.** Design sweeps and UQ for LOX/H2 and LOX/CH4 combustion that need
  10^4–10^6 reacting runs and tolerate percent-level error. Also the exact case
  that motivated the bet.
- **State and units.** Input (T [K], p [Pa], Y_1..Y_ns [–], Δt [s]) at constant
  p; output ΔY in ker E after projection (I − E⁺E); T recovered by Newton on the
  enthalpy constraint; positivity and renormalization. Δt ∈ {1e-7, 1e-6, 1e-5}
  s. Envelope to freeze: T0 800–2500 K, p 1–100 atm (extend toward 300 atm for
  engine conditions later), φ 0.3–3.0.
- **Mechanism (decision).** `h2o2.yaml` (10 species, GRI-3.0 H2/O2 subset) is
  the reproducible default and matches the literature. For rocket conditions
  substitute a high-pressure mechanism (Burke 2012 or Kéromnès 2013) and use
  O2 rather than air as oxidizer. Pick one before generating truth.
- **Data.** Cantera 3.2 CVODE, rtol 1e-8 / atol 1e-15 for truth, cold-start.
  Manifold-aware sampling from 0-D ignition, 1-D freely propagating flames
  (Cantera `FreeFlame`), counterflow; constrained augmentation transverse to the
  manifold (DFODE-Kit's `random-local-combustion-v1` is the reference).
  Splits hold out whole φ bands and p bands, never neighbouring samples. Cost:
  minutes per 10^6 samples (section 2.3).
- **Model.** Residual MLP, 0.1–2M parameters, per-species Box–Cox/log
  transform plus standardization, hard projection, enthalpy temperature,
  Mahalanobis and ensemble-disagreement gates with CVODE fallback. No
  pretrained weights: this track must be allowed to win without transfer.
  Optional cheap transfer arm: initialize the H2 model from a GRI-3.0-trained
  model (cross-mechanism transfer inside the chemistry modality).
- **Baselines (all on the same axes).** (a) cold-start CVODE at 1e-6/1e-12 and
  1e-8/1e-15; (b) ISAT-lite: kNN tabulation with error control, the practical
  incumbent; (c) Stiff-PINN with QSSA, reference implementation and authors'
  hyperparameters; (d) vanilla PINN, to reproduce the original divergence with
  the conditioning measured; (e) a neural-ODE learned source term
  (ChemNODE-style); (f) DFODE-Kit's own trained model if its Cantera-only
  path runs without OpenFOAM.
- **Metrics.** Ignition delay, T and log-relative species profiles, ‖EΔY‖∞,
  enthalpy drift, fallback rate, first-unacceptable time, over 10^3–10^5-step
  a posteriori 0-D rollouts; laminar flame speed and profiles in a small
  operator-split 1-D reaction–diffusion solver (to be written; ~200 lines
  NumPy/PyTorch, Strang split, chemistry substep swappable); per-cell wall time
  cold-start CPU and batched GPU; end-to-end on the 1-D solver with chemistry
  share f measured; break-even count.
- **Compute.** CPU for truth; RTX 3060 for training (minutes to hours); no cloud.
- **Preregistered go/no-go.** Go if, on held-out φ and p bands: ignition delay
  within 2%, T within 1%, fallback < 5%, batched surrogate ≥ 50x cheaper per
  cell than cold CVODE at production tolerance at that error, 1-D flame speed
  within 3%, and Stiff-PINN and ISAT-lite reported on the same table. No-go if
  manifold sampling plus augmentation cannot reach these, or if gate deferral
  exceeds d_break = (1 − g − r)/(1 − r); record either as the chemistry bet
  losing at this envelope.
- **Strongest reason it may be pointless.** DFODE-Kit already ships this
  pipeline under GPL-3. The new content is only: the adversarial baseline set
  including Stiff-PINN and a tabulation incumbent, hard projection and
  enthalpy temperature instead of penalties, gating economics measured, and a
  propulsion envelope. Without a real CFD host (DeepFlame/OpenFOAM in Docker),
  no end-to-end claim beyond the 1-D solver is possible.

### E2 — Pretraining-source ablation on one Well dataset (research track)

- **Value.** Settles the weight-reuse half of the bet with the comparison
  nobody has published: natural-video initialization vs physics-pretrained
  initialization vs scratch, at matched architecture and budget, with FNO/TFNO
  and U-Net as specialist baselines.
- **State and units.** The Well `turbulent_radiative_layer_2D`: 90
  trajectories (9 cooling times × 10 seeds), 101 steps, 384×128, fields
  density, pressure, velocity (2), Athena++, periodic in x, zero-gradient in y,
  6.9 GB. Dimensionless units; report per-field VRMSE after inverse
  normalization.
- **Data.** `pip install the_well` (1.2.0 resolves in the Surrogates venv).
  Split by t_cool value: hold out 2 of 9 values, never frames.
- **Arms.** (i) Tokeniser arm (story 13, narrowed): Cosmos-1.0 CV8x8x8
  continuous tokenizer channel-adapted vs its scratch twin vs a MAGVIT-2-style
  tokeniser pretrained on other Well sets (Sotoudeh recipe); record every
  retained, inflated, replaced, frozen, trained parameter. (ii) Dynamics arm:
  GPhyT-S (9.2M) or PDE-Transformer-S pretrained vs scratch vs TFNO
  (neuraloperator) vs U-Net vs persistence, 3 seeds. Walrus 1.3B only after
  LoRA memory is measured on 12 GB.
- **Metrics.** Per-field VRMSE in physical units, spectra, rollout divergence
  time over ≥ 50 steps, mass and momentum integrals, boundary violations,
  held-out t_cool. No wall-time claim: Athena++ is not available locally, so
  this is surrogate-to-surrogate by construction.
- **Compute.** RTX 3060 12 GB: GPhyT-S and PDE-Transformer-S fit comfortably;
  Cosmos-1.0 tokenizer training memory and Walrus LoRA memory must be measured
  from a small run before scheduling. Whether the archived Cosmos-1.0 tokenizer
  is still downloadable must be verified.
- **Preregistered go/no-go.** Pretrained init beats scratch by ≥ 15% VRMSE at
  10k steps on 3/3 seeds without losing spectral or conserved structure →
  weight reuse holds for that modality. If natural-video init < cross-domain
  physics init (Sotoudeh's −19%), amend the thesis to "pretrain on physics,
  not video." If natural video ≤ scratch, close the Cosmos line.
- **Strongest reason it may be pointless.** If natural video adds nothing over
  cross-domain physics, the answer ("use Walrus/GPhyT") is already known, and an
  astrophysical cooling layer has no propulsion value in itself. It is a
  science result, not an engineering one.

### E3 — LLM agent as the surrogate engineer (authoring track)

- **Value.** Where a text-pretrained model demonstrably beats hand-built PINN
  and FNO workflows: producing, from a story, the envelope, sampling plan,
  Cantera data generation, training, admissibility checks, adversarial
  evaluation, and report. Matches the charter's AI-first properties and PDD.
- **Test.** The agent reproduces E1's benchmark table from the story and the
  contract without hand-edited code, with every number reproducible.
- **Baseline.** DeepFlame 2.0's DFODE-kit Trainer agent is prior art; ours must
  be judged on whether its output passes E1's go/no-go, not on demo quality.
- **Strongest reason it may be pointless.** It is E1 with a wrapper unless the
  reproducibility and pass criteria are strict. Runs only after E1 exists as
  the reference.

## 5. Force ranking and the single next action

1. **E1** — highest engineering value, cheapest truth, incumbent measured,
   PINN/FNO-class baselines can be run honestly, no downloads.
2. **E2** — the real open science question, narrowed; 3060-feasible with the
   new small physics-pretrained weights; second because it cannot produce a
   solver-speedup claim.
3. **E3** — after E1.

**The one next research action that needs no model download:** freeze the E1
benchmark contract. Concretely: (a) choose mechanism and oxidizer (H2/air
`h2o2.yaml` for reproducibility, or a high-pressure H2/O2 mechanism for engine
relevance); (b) fix the envelope, Δt set, tolerances, splits, margins above;
(c) generate the truth set and publish the baseline table with cold-start
CVODE, ISAT-lite, and the Stiff-PINN reproduction *before* any surrogate is
trained. That table is the thing every later claim is measured against.

## 6. Three decisions only Ernest can make

1. Mechanism and oxidizer for E1 (air-breathing reproducibility vs rocket
   relevance).
2. Whether an end-to-end CFD host is in scope this quarter (DeepFlame/OpenFOAM
   in Docker) or whether the 1-D operator-split solver is the end-to-end proxy.
3. Whether E2 proceeds on the astrophysical Well dataset (cheap, comparable
   to the literature) or waits for a reacting-flow field dataset the repo does
   not yet have.


### 6.1 Recommendations (added later the same day, after Ernest asked)

**Decision 1 — mechanism and oxidizer: CH4/O2, no nitrogen, Raptor-class envelope.**
SpaceX's flagship engine is LOX/methane full-flow staged combustion at roughly
300 bar with an oxidizer-to-fuel mass ratio near 3.6; Merlin is LOX/RP-1, whose
kerosene surrogates run to hundreds of species and a variable fuel
composition. Methane is both the most SpaceX-relevant and the most tractable
choice. Concretely:

- **Primary mechanism: FFCM-1** (Stanford, Smith/Tao/Wang 2016; 38 species,
  291 reactions; MIT-style academic release as CHEMKIN files, convert with
  `python -m cantera.ck2yaml`, which is present in the Surrogates venv).
  Optimized with uncertainty quantification against a broad target set; its
  reduced-model validation literature covers 1–120 atm, 1000–2500 K,
  φ 0.6–1.4. Nothing public is validated at 300 bar; every rocket CFD group
  extrapolates, and so will we, saying so.
- **Reproducibility cross-check: the GRI-3.0 C/H/O submechanism** built from
  the `gri30.yaml` that ships with Cantera by dropping every N-containing
  species except inert N2. Measured today: 36 species, 219 reactions (full
  GRI-3.0 is 53/325). Ships with Cantera, so anyone can rerun it.
- **Week-one smoke test: H2/O2** (`h2o2.yaml`, 10 species). Hydrogen chemistry
  is a subset of any methane mechanism, so this is also the natural source
  arm for the cross-mechanism transfer test in E1.
- **Envelope, main chamber first:** p 100–350 bar, φ 0.5–2.0 (mixing layers
  around the O/F 3.6 operating point, which is φ = 1.11), T 700–4000 K
  (ideal-gas adiabatic flame temperature at 300 bar, O/F 3.6, 800 K inlet
  computed today as 3958 K), Δt {1e-8, 1e-7, 1e-6} s. At 300 bar the fast
  eigenvalue is roughly 10^10 s^-1 and rocket LES steps are 10–100 ns, so the
  Δt set shifts down two decades from the atmospheric benchmark in section 2.
  Preburner regimes (oxidizer-rich φ ≈ 0.1–0.2, fuel-rich φ ≈ 3–6) are a second
  envelope, not the first.
- **Equation of state: ideal gas first, Peng–Robinson as a later arm.** Cantera 3
  supports Peng–Robinson and Redlich–Kwong with kinetics. At flame temperatures
  the compressibility factor is near one; near the injector it is not. The
  ideal-gas simplification is deliberate and must be stated in the contract.

**Decision 2 — both, sequenced: the 1-D solver first, DeepFlame second, and
DeepFlame's first job is measurement, not deployment.**

- The 1-D operator-split reaction–diffusion solver (Strang split, chemistry
  substep swappable, Cantera for transport and thermo, ~200–300 lines) is on
  the critical path for the E1 go/no-go: a posteriori flame speed against
  Cantera `FreeFlame`, ignition in a gradient, chemistry runtime share f,
  fallback economics. Week 1–2.
- DeepFlame 2.0 is the end-to-end host, but there is no official Docker image
  or Dockerfile (checked the GitHub README and the v1.5 install docs on
  2026-09-01). Its documented install is OpenFOAM-7, LibCantera 2.6 through
  conda, Python 3.8, PyTorch cu118. We write our own Dockerfile the way the
  CLIImage and Isaac stacks were done; budget one to two days. Its DNN
  chemistry interface expects DFODE-kit-format `.pt` models, so our surrogate
  conforms to that interface rather than re-plumbing OpenFOAM.
- The first DeepFlame task is to run a stock methane tutorial case with CVODE
  and profile the chemistry share f. That single number sets the Amdahl
  ceiling 1/((1−f)+f/S) and decides whether the end-to-end claim is worth
  pursuing at all, before any surrogate is trained. Thümmler–Kuroda's 16.9%
  is the cautionary case.

**Decision 3 — do not wait, and do not use the astrophysical dataset.**

- How `turbulent_radiative_layer_2D` was made: it is simulation output, not
  measurement. Athena++ runs of a Kelvin–Helmholtz-driven mixing layer between
  hot and cold gas with radiative cooling, after Fielding et al. 2020, with the
  cooling time swept over 9 values and 10 seeds each. Every dataset in The
  Well (15 TB) is numerical simulation; the Polymathic group's only contact
  with measured data is the Walrus lab Rayleigh–Taylor transfer paper.
- A subtler reason to avoid it: Walrus, GPhyT, and PDE-Transformer were
  pretrained on The Well or overlapping corpora, so fine-tuning any of them on
  a Well dataset is in-distribution and contaminates the transfer comparison
  in E2. The ablation needs a dataset none of the arms has seen.
- Two reacting-flow sources satisfy that and are propulsion-relevant:
  1. **BLASTNet** (Stanford, Ihme group): 4.8 TB of compressible turbulent DNS
     in <100 GB subsets on Kaggle, including reacting forced HIT, H2/CH4
     turbulent jet flames, and premixed H2-air and NH3-H2-air flames. 3-D,
     CC BY-NC-SA 4.0 (noncommercial). Fields and mechanisms per subset must be
     read off the individual Kaggle pages before choosing.
  2. **Self-generated with DeepFlame** once decision 2 lands: 2-D laminar and
     turbulent CH4/O2 flames with the E1 mechanism. Slower to obtain, but the
     truth solver is then rerunnable locally, which is the only way E2 can ever
     make a solver-speedup claim rather than a surrogate-to-surrogate one.
- Sequence: E2 starts in month 2 with a BLASTNet reacting subset for the
  ablation, and moves to DeepFlame-generated CH4/O2 fields when they exist.
  Nothing in E2 blocks E1.

## 7. State

- **Track A (E1): READY FOR BENCHMARK CONTRACT.** Candidate, data plan, incumbent
  cost, baselines, metrics, and margins are fixed above pending decision 1.
  Not a user story yet: no independently testable behaviour exists until the
  contract is frozen and the baseline table is published.
- **Track B (E2): NOT READY.** The candidate list changed (physics-pretrained
  weights now dominate Cosmos3-Edge), no Well slice is downloaded, and training
  memory on the 3060 is unmeasured.
- **Track C (E3): NOT READY** by construction; depends on E1.

## Sources

- Sotoudeh et al., arXiv:2603.05598 — https://arxiv.org/abs/2603.05598
- GPhyT, arXiv:2509.13805v4 — https://arxiv.org/abs/2509.13805 ; code https://github.com/FloWsnr/General-Physics-Transformer ; weights https://huggingface.co/flwi/Physics-Foundation-Model
- PDE-Transformer, arXiv:2505.24717 — https://huggingface.co/thuerey-group/pde-transformer
- Walrus, arXiv:2511.15684 — https://huggingface.co/polymathic-ai/walrus ; lab transfer arXiv:2606.01470
- Cosmos3-Edge — https://huggingface.co/nvidia/Cosmos3-Edge
- PhysiX — https://github.com/arshka/PhysiX ; arXiv:2506.17774
- Breakeven complexity, arXiv:2605.15399 — https://arxiv.org/abs/2605.15399
- No Free Lunch in Flow Surrogates, arXiv:2607.23667 — https://arxiv.org/abs/2607.23667
- Thümmler and Kuroda, arXiv:2608.23075 — https://arxiv.org/abs/2608.23075
- Predictivity and Utility, arXiv:2604.20061 — https://arxiv.org/abs/2604.20061
- Stiff-PINN, arXiv:2011.04520 — https://github.com/DENG-MIT/Stiff-PINN
- Liu et al., arXiv:2402.00795 ; LLM-ODE arXiv:2603.20910
- DFODE-Kit — https://github.com/deepflame-ai/DFODE-kit ; DeepFlame 2.0 — https://blogs.deepmodeling.com/DeepFlame2.0_28_1_2026
- The Well `turbulent_radiative_layer_2D` — https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/
- McGreivy and Hakim, arXiv:2407.07218
