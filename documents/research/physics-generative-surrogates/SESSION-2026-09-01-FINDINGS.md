# Session findings, 2026-09-01: from "fine-tune an LLM for physics" to three runnable experiments

**Audience:** any AI agent or harness picking this up cold. Read this file first.
**Repository:** `/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/Propulsion`
(the same files are reachable as `/home/propdev/.openclaw/workspace/workspace2/repos/Propulsion`;
`workspace2` is a symlink, not a second checkout).
**Branch:** `feat/corpus-user-stories`. Nothing from this session is committed.
**Owner:** Ernest Yeung. Do not commit to `master` or `main`; feature branches are fine.

## 1. The question, and what it became

Ernest's question: can an LLM, or a fine-tuned open-weights model, become a
numerical simulator for physics, and beat physics-specific architectures such
as physics-informed neural networks (PINNs) and Fourier neural operators (FNOs)?
The motivating failure was a PINN on stiff combustion chemistry that diverged.

The session's answer, with evidence in section 3:

1. "Beat PINN and FNO" is the wrong finish line. Beating a learned baseline on
   held-out error is an ML-versus-ML claim; McGreivy and Hakim (2024) showed
   79% of such claims used weak baselines. The scoreboard that matters is
   **wall time at matched physical error against the tuned classical solver,
   end to end, with the break-even count**. PINN and FNO become mandatory
   baselines on that table, not the target.
2. The text-pretrained LLM is not the numerical kernel. Text shares no
   structure with field arrays or thermochemical state vectors, and the one
   repository that tried it showed a 700K-parameter scratch model beating a
   410M pretrained one. The LLM's honest role is **the engineer**: it writes
   the envelope, sampling plan, training code, evaluation, and report.
   DeepFlame 2.0 shipped exactly that as an agent in January 2026.
3. Where a transformer and open weights do belong is the **field surrogate**:
   transformers pretrained on physics simulations (GPhyT, PDE-Transformer,
   Walrus, all MIT, all fit the RTX 3060 at small size) or on natural video
   (the Cosmos-1.0 tokenizer). The open scientific question nobody has
   published: does natural-video pretraining help as much as physics
   pretraining, at matched architecture and budget?
4. The **stiff-chemistry substep** is where a learned model can beat the
   classical solver on the real scoreboard this quarter, and it needs no
   transformer and no pretrained weights. A small supervised network learns
   the finite-time flow map of the chemistry ODE, with conservation enforced
   by projection, temperature recovered from enthalpy, and a gate that falls
   back to CVODE off-manifold.

Those became three experiments, named E1 (chemistry), E2 (field surrogate),
E3 (agent). Earlier documents call E1 "Track A" or "Track 1" and E2 "Track B"
or "Track 2"; `documents/CHARTER.md` uses "Track A–E" for something unrelated.
Use E1/E2/E3 from now on.

## 2. Where the AI is, experiment by experiment

| | E1 chemistry | E2 field surrogate | E3 agent |
|---|---|---|---|
| What is simulated | 0-D ignition and 1-D flames with Cantera CVODE; later DeepFlame 3-D | 2-D/3-D reacting flow fields: BLASTNet first, DeepFlame-generated later | nothing new; reuses E1 |
| What the network learns | the chemistry flow map: state now to state one step later | next field frames from the last few, autoregressively | to produce the whole E1 pipeline from a story |
| Architecture | residual MLP, 0.1–2M parameters | transformer over space-time patches: GPhyT / PDE-Transformer / Walrus | Claude or a local open-weight LLM via OpenClaw |
| Open weights | none, deliberately (optional chemistry-to-chemistry init only) | physics-pretrained, plus a natural-video arm, each vs the same net from scratch | text-pretrained |
| Where "beat" is scored | cold-start CVODE on wall time at matched error; Stiff-PINN and tabulation on the same table | FNO/TFNO, U-Net, scratch twin on rollout error over a reacting flow none of them saw | a hand-built pipeline: does the agent's output pass E1's go/no-go |

Not used, and why: text LLMs as kernels (no shared structure); Cosmos3-Edge
(RGB, diffusion, action-conditioned; physics-pretrained models a tenth its size
now exist); PhysiX checkpoints (never released); custom CUDA training
(premature until a model proves useful).

## 3. Evidence gathered (primary sources, all checked 2026-09-01)

Landscape, new since the previous handoff:

- **Sotoudeh et al., arXiv:2603.05598** (Polymathic, ICLR 2026 AI&PDE workshop).
  Tokeniser pretraining on Well data: in-domain −64% VRMSE at 10.5k steps,
  cross-domain physics −19%, natural video **not tested**. This partly answers
  story 13 and narrows E2 to "natural video vs physics pretraining."
- **Physics-pretrained open weights** that fit a 12 GB GPU: GPhyT S/M/L at
  9.2M/112M/385M (MIT; 4 input frames at 256×128; neural differentiator plus
  Forward Euler; "more than 7x" lower next-step NMSE than DPOT), PDE-Transformer
  at 33M–701M (MIT), Walrus at 1.3B (MIT; transferred to lab Rayleigh–Taylor
  with ≤3 DNS realizations, arXiv:2606.01470).
- **Cosmos3-Edge**: 4B, released 2026-07-20, BF16 only, RGB/text/action
  interfaces, no field interface, no verified fine-tune recipe. Demoted.
- **PhysiX**: no checkpoints on GitHub as of today; depends on archived
  Cosmos-1.0 weights; recipe uses 8 GPUs.
- **The 2026 evaluation literature** converged on the wall-time-at-matched-error
  scoreboard: Breakeven complexity (arXiv:2605.15399); No Free Lunch in Flow
  Surrogates (arXiv:2607.23667; no architecture won both regimes); Thümmler and
  Kuroda (arXiv:2608.23075), a controlled **negative** result where a 1.8M MLP
  5.8x cheaper per call reached parity at best because the substep was 16.9% of
  runtime and the Mahalanobis gate deferred 96.8–99.7% of cells; break-even
  deferral d_break = (1−g−r)/(1−r). Predictivity and Utility
  (arXiv:2604.20061): spectral bias, low-dimensional manifolds, weather is a
  sweet spot.
- **Text LLM and numerics**: Liu et al. arXiv:2402.00795 (LLaMA-2 in-context
  dynamical systems, zero-shot, slow, no matched-compute specialist baseline);
  LLM-ODE arXiv:2603.20910 (symbolic discovery, authoring). DeepFlame 2.0
  (2026-01-28) DFODE-kit Trainer agent: LLM as engineer of a chemistry surrogate.
- **Stiff-PINN** (Ji et al., arXiv:2011.04520, DENG-MIT/Stiff-PINN): PINNs work
  on stiff kinetics only after QSSA removes the stiffness. This is the strongest
  PINN baseline and must be run with the authors' code.
- **DFODE-Kit** (deepflame-ai/DFODE-kit, GPL-3): Cantera/CVODE labels, HDF5,
  augmentation preset `random-local-combustion-v1`; full workflow needs
  OpenFOAM + DeepFlame; isolated labeling does not.
- **BLASTNet** (Stanford, Ihme group): 4.8 TB compressible turbulent DNS in
  <100 GB Kaggle subsets, including reacting forced HIT, H2/CH4 jet flames,
  premixed H2-air and NH3-H2-air flames. 3-D. CC BY-NC-SA 4.0.
- **DeepFlame 2.0**: no official Docker image or Dockerfile. Documented install:
  OpenFOAM-7, LibCantera 2.6, Python 3.8, PyTorch cu118. DNN chemistry via
  DFODE-kit-format `.pt` models.
- **The Well `turbulent_radiative_layer_2D`** is Athena++ simulation output
  (Kelvin–Helmholtz mixing layer with radiative cooling, 9 cooling times × 10
  seeds, 384×128, 101 steps, 6.9 GB). Nothing in The Well is measured. Walrus,
  GPhyT, and PDE-Transformer were pretrained on The Well or overlapping
  corpora, so fine-tuning them on it contaminates a transfer comparison.

Ledger corrections: GPhyT v4 abstract says "more than 7x" (next-step NMSE vs
DPOT); the earlier "29x" was not found. Section C of
`EVIDENCE-AND-OPEN-QUESTIONS.md` lacked GPhyT/PDE-Transformer/Walrus.

## 4. Measured this session

Environment created: `Surrogates/.venv` (uv, Python 3.13.3) with Cantera 3.2.0,
NumPy 2.5.2, SciPy 1.18.1. Script: `Surrogates/stiffness_benchmark.py`.
Raw results: `Surrogates/results/stiffness_benchmark.json`.

Stiffness ratio ς = max|Re λ| / min_active|Re λ| of the chemistry Jacobian,
constant-pressure ignition, fuel with air:

| Mechanism | T0 [K] | p [atm] | φ | τ_ign [s] | ς worst (induction) | ς at ignition |
|---|---|---|---|---|---|---|
| h2o2 (10 species) | 1000 | 1 | 1.0 | 3.1e-4 | 3.6e8 | 2.5e3 |
| h2o2 | 1200 | 1 | 1.0 | 4.5e-5 | 8.6e6 | 1.4e3 |
| h2o2 | 1500 | 1 | 1.0 | 1.3e-5 | 2.7e5 | 8.7e2 |
| h2o2 | 1200 | 10 | 1.0 | 6.4e-5 | 8.6e6 | 1.0e3 |
| gri30 (53 species) | 1400 | 1 | 1.0 | 3.4e-3 | 7.6e12 | 2.0e5 |
| gri30 | 1400 | 10 | 1.0 | 5.0e-4 | 1.5e12 | 1.0e5 |

The ς ~ 10^8 (H2) and ~10^12 (hydrocarbon) figures in `SourceTermSurrogate.tex`
are reproduced but are induction-period values; at ignition ς is 10^3 and 10^5.
Report both with the trajectory position attached.

Cost of the incumbent (Cantera CVODE, single core, Python API):

| Mechanism | Δt [s] | rtol/atol | warm µs/call | **cold µs/call** |
|---|---|---|---|---|
| h2o2 | 1e-6 | 1e-6/1e-12 | 21 | **128** |
| h2o2 | 1e-6 | 1e-8/1e-15 | 40 | 185 |
| gri30 | 1e-6 | 1e-6/1e-12 | 6 | **1335** |
| gri30 | 1e-6 | 1e-8/1e-15 | 10 | 1912 |

Cold start (reset state, reinitialize integrator, advance) is what an
operator-split CFD code does per cell per step and is the denominator a
surrogate must beat. Truth for 10^6 samples costs minutes.

Also computed: the GRI-3.0 C/H/O submechanism (drop nitrogen chemistry, keep
inert N2 and Ar) is 36 species / 219 reactions; at O/F 3.6 by mass CH4/O2 the
equivalence ratio is 1.11; ideal-gas adiabatic flame temperature at 300 bar
with 800 K inlet is 3958 K.

## 5. Decisions recommended (Ernest has not yet confirmed)

1. **Mechanism and oxidizer:** CH4/O2, no nitrogen, Raptor-class envelope
   (p 100–350 bar, φ 0.5–2.0, T 700–4000 K, Δt {1e-8, 1e-7, 1e-6} s).
   Primary mechanism FFCM-1 (38 species / 291 reactions; CHEMKIN files from
   Stanford, convert with `python -m cantera.ck2yaml`). Reproducibility check:
   the GRI-3.0 C/H/O submechanism. Week-one smoke test: `h2o2.yaml`.
   Ideal gas first, Peng–Robinson later. Nothing public is validated at
   300 bar; say so.
2. **Hosts:** both. The 1-D operator-split solver first (critical path for the
   go/no-go), DeepFlame 2.0 second via our own Dockerfile, and DeepFlame's
   first job is to profile the chemistry share of wall time f, which sets the
   Amdahl ceiling 1/((1−f)+f/S).
3. **Field data:** do not wait, do not use the astrophysical Well set. Use a
   BLASTNet reacting subset for the E2 ablation (out of every arm's
   pretraining corpus), then DeepFlame-generated CH4/O2 fields, which make a
   solver-speedup claim possible because the truth solver is rerunnable.

## 6. Files this session created or changed

Created:
- `documents/research/physics-generative-surrogates/CANDIDATE-EXPERIMENTS-2026-09-01.md`
  (the research memo: audit, measurements, E1–E3 with preregistered go/no-go,
  recommendations in §6.1, state in §7)
- `documents/research/physics-generative-surrogates/SESSION-2026-09-01-FINDINGS.md` (this file)
- `documents/research/physics-generative-surrogates/RESULTS-LEDGER.md` (claim → evidence → artifact)
- `documents/research/physics-generative-surrogates/tasks/` (task board and briefs)
- `Surrogates/README.md`, `Surrogates/stiffness_benchmark.py`, `Surrogates/results/stiffness_benchmark.json`
- `Surrogates/.venv` (gitignored)

Modified:
- `documents/research/physics-generative-surrogates/README.md` (reading order)
- `documents/research/physics-generative-surrogates/EVIDENCE-AND-OPEN-QUESTIONS.md` (addendum F)

Untouched on purpose: `documents/stories/12-*`, `13-*`, `14-*`.
Pre-existing uncommitted edits by Ernest: `documents/PHYSICS-ML-BET.md`,
`documents/stories/README.md`.

## 7. Standing rules for any agent working here

Adversarial discipline (story 09, ledger §E):
- Name the baseline behind every "x times" number; separate one-step error,
  rollout error, and wall-clock.
- Never change a baseline's tolerance or hyperparameters after seeing a
  surrogate result. Never tune on the held-out split.
- Record which pretrained parameters survived any interface change.
- Preserve negative results with the same artifacts as positive ones.
- A visually plausible rollout is not numerical evidence.
- Append every claim to `RESULTS-LEDGER.md` with the artifact path.

Ernest's conventions (from memory, verified this session):
- Python only through uv virtual environments. Never system pip, never conda.
  Surrogates venv: `cd Surrogates && source .venv/bin/activate && uv pip install <pkg>`.
- Never commit to `master`/`main`. Feature branches may be pushed.
- Never put `Data/Private/` paths in anything committed; datasets go under
  `/media/propdev/Expansion/openclaw/.openclaw/workspace/Data/Public/datasets/`.
- GPUs: nvidia-smi index 0 is a GTX 980 Ti (6 GB, Maxwell, **unsupported by
  CUDA 13 wheels**), index 1 is the RTX 3060 (12 GB). CUDA device ordering
  differs from nvidia-smi; verify with `torch.cuda.get_device_name(0)` and pin
  `CUDA_VISIBLE_DEVICES` to the 3060.
- Docker 29.1 with the nvidia runtime is installed on the host.
- No Kaggle API token exists on this machine; BLASTNet downloads need Ernest
  to place `~/.kaggle/kaggle.json`.
- Do not download multi-billion-parameter checkpoints or rent cloud GPUs
  without an explicit instruction.

## 8. State and next action

- E1: READY FOR BENCHMARK CONTRACT, pending decision 1.
- E2: NOT READY (dataset and candidate list changed; memory unmeasured).
- E3: NOT READY by construction; needs E1 as reference.

Next action: `tasks/README.md`, starting with TASK-00 and TASK-01.
