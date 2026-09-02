# Session handoff: pretrained generative models as numerical surrogates

**Prepared:** 2026-09-01  
**Repository:** `/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/Propulsion`  
**Branch observed:** `feat/corpus-user-stories`  
**Maturity:** pre-story research

## 1. What Ernest is trying to determine

The long-term goal is to determine whether knowledge and representations in an
existing open-weights generative model can be transferred into physics
simulation work, instead of inventing a new neural architecture or training a
large model from scratch.

The original intuition was intentionally contrarian:

- PINNs, FNOs, and other physics-specific architectures often succeed on clean
  academic benchmarks but may fail on messy engineering regimes.
- Combustion chemical kinetics supplied a concrete failure: reaction rates and
  species concentrations span many orders of magnitude, and residual-based PINN
  training became unstable.
- A broadly pretrained model may already contain reusable structure, just as a
  person can transfer general intellectual machinery into a new scientific
  domain after concentrated training.
- Therefore, test fine-tuning or other transfer from open weights before
  committing to a bespoke “physics AI” architecture.

The goal is not to defend that thesis indefinitely. It is to turn it into
experiments that can falsify it.

## 2. The phrase “LLM numerical generator” currently hides four different jobs

### 2.1 Physics authoring and research assistance

A text-pretrained LLM can write derivations, code, mechanism files, reward or
cost functions, solver configurations, tests, and experiment plans. Text and
code pretraining directly overlap this target. This is already useful, but it
does not make the LLM a numerical simulator.

The term “Large Physics Models” is used in the literature for broad
foundation-model systems tailored to physics research. It includes symbolic,
literature, data-analysis, and collaborative research capabilities; it should
not be used as a synonym for a PDE surrogate.

### 2.2 Stiff-chemistry substep surrogate

This is a supervised map inside an otherwise classical solver:

\[
  (T, p, Y_1,\ldots,Y_{n_s},\Delta t)
  \longmapsto
  (T',Y'_1,\ldots,Y'_{n_s}).
\]

The network replaces a call to CVODE, Radau, or another stiff chemistry
integrator. Transport, boundary conditions, the pressure solve, turbulence, and
the overall CFD loop remain classical. This is the lowest-risk ML route in the
actual combustion domain.

It is not naturally an LLM problem. A relatively small supervised model may be
the best answer, and pretrained weights may add no value. The transfer thesis
must be allowed to lose here.

### 2.3 Full-field rollout surrogate

This predicts future grid fields from past grid fields and parameters:

\[
  (u_{t-k:t},\theta,\text{BCs}) \longmapsto u_{t+1:t+m}.
\]

It is in the same broad category as neural operators, learned emulators, and
physics foundation models. Here video pretraining is plausibly relevant because
both targets are spatiotemporal arrays with moving, deforming, and flowing
structure. This is where PhysiX, GPhyT, and a possible Cosmos3 adaptation belong.

### 2.4 Visual world generation

A world model can generate plausible future RGB frames or infer robot actions.
That is useful for robotics, autonomous systems, Isaac simulation workflows,
and qualitative scenario generation. It is not a numerical surrogate unless it
also preserves physical variables, units, admissibility, conservation, and
long-horizon error within declared tolerances.

## 3. The refined thesis

The defensible thesis after the discussion is:

> Reuse a standard pretrained generative architecture and its weights when the
> pretraining modality shares structure with the target. Use text pretraining
> for physics authoring and symbolic work; test video/world-model pretraining for
> spatiotemporal field surrogates. Do not assume that text pretraining transfers
> to raw numerical trajectories, and do not assume that a larger model beats a
> small specialist.

Two claims must remain separate:

1. **Architecture reuse:** a standard transformer or current world-model stack
   may be preferable to inventing a novel architecture.
2. **Weight reuse:** pretrained initialization may outperform the same
   architecture trained from scratch.

Evidence for architecture reuse is not evidence for weight reuse. Every transfer
experiment needs an architecturally identical scratch control.

## 4. What the two hackathon repositories actually established

The motivating repositories were:

- `/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/jsjung00/locomotion-language-model`
- `/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/ninjaa/web-mujoco-gym`

They were useful prompts but are not evidence for an LLM numerical surrogate.

### 4.1 `locomotion-language-model`

This is a Decision Transformer-style HalfCheetah experiment with a frozen
Pythia-410M trunk. Continuous states are passed through `inputs_embeds`; the
normal token embedding and language-model head are replaced. No text tokenizer
or textual physics knowledge participates in the task.

Its reported small from-scratch GPT-2 baseline outperformed the frozen Pythia
configuration. That does not prove that all transfer fails: the experiment
changed the semantic interface, used low-dimensional continuous control, and
compared different training arrangements. It does prove that this repository
must not be cited as support for the transfer thesis.

Training is supervised action regression on an offline dataset, not online
reinforcement learning. Return-to-go is a conditioning variable; that alone
does not make the training loop RL.

### 4.2 `web-mujoco-gym`

Claude generates reward functions and suggests hyperparameters, which is useful
evidence for LLM-assisted objective authoring. The inspected main implementation
did not establish working RL: the summarized update signal collapsed toward
zero, and the policy action path did not control the simulator path that was
being evaluated. The repository itself described the demonstration as “Faux
RL.”

These are two-day hackathon artifacts. The findings are about evidentiary scope,
not about the people who built them.

## 5. Why the PINN objection remains serious

For stiff chemical kinetics, the source Jacobian contains modes separated by
many decades. A residual loss of the form

\[
  \left\|\frac{d\phi_\vartheta}{dt}-S(\phi_\vartheta)\right\|^2
\]

inherits the conditioning of the source term. Fast modes dominate the loss and
slow heat-release modes can contribute gradients beneath numerical noise. This
is the same structural stiffness that motivates implicit integration and
Jacobian-based methods classically.

The local write-up
`LaTeXandpdfs/SourceTermSurrogate.tex` develops the alternative: learn the
finite-time chemistry flow map rather than the vector field or a global PINN
solution. Stable fast directions contract under the flow map, so precisely the
modes that make the vector field violent may become easier to approximate after
a finite time step.

The proposed engineering constraints are:

- predict increments or a finite-time map, not raw instantaneous reaction rates;
- transform and standardize each species separately because abundances span
  many decades;
- enforce element and total-mass conservation by projection rather than only a
  tunable penalty;
- obtain temperature from the thermodynamic enthalpy constraint when possible;
- enforce positivity and composition normalization;
- reject out-of-envelope states and fall back to the classical integrator;
- evaluate a posteriori in coupled trajectories, not only on held-out one-step
  samples;
- compare with the practical incumbent—ISAT, flamelet/manifold methods, or
  another tabulation approach—not only naive CVODE.

DFODE-Kit is the most concrete starting implementation identified. It is an
open package for learned stiff-chemistry integration with DeepFlame/OpenFOAM
integration and reports both isolated-chemistry and coupled-CFD acceleration.
That result supports trying a source-term surrogate; it does not support using a
large pretrained model for it.

## 6. Evidence for transferred full-field models

### 6.1 PhysiX

PhysiX (`arXiv:2506.17774`) is a 4.5B autoregressive generative model for physics
fields. Its important result for this program is the pretrained-versus-scratch
ablation: it adapts a Cosmos video tokenizer and Cosmos-1.0 autoregressive model,
and reports benefits from pretrained initialization.

That is evidence that video/generative pretraining can transfer to simulation
fields. It is not evidence that the surrogate beats its classical data-generating
solver. The paper’s comparisons are principally against other learned models;
the classical solver defines the ground truth.

PhysiX should be treated as a proof of possibility and a source of experimental
techniques, not as a frozen recipe we must reproduce exactly. Its public code
references:

- `nvidia/Cosmos-1.0-Tokenizer-DV8x16x16`
- `nvidia/Cosmos-1.0-Autoregressive-4B`

No official, ready-to-use fine-tuned PhysiX checkpoint was found on Hugging Face
during the session. Recheck before relying on that statement.

### 6.2 GPhyT and other physics foundation models

The General Physics Transformer (`arXiv:2509.13805`) reports broad simulation
pretraining, cross-system generalization, and long rollouts. Its headline
comparison is against specialized learned architectures, not a tuned classical
solver. The current arXiv abstract reports “up to 29x”; an earlier session note
said “7x,” so the next session must identify the exact paper version, metric,
and baseline before repeating either number.

PDE-FM, SPUS, PhysiX, GPhyT, and newer work such as the July 2026 “Physics
Transformer” are candidate landscape items. None should be accepted from an
abstract alone. The next research pass must inspect open weights, code,
licenses, training data, rollout metrics, and classical-solver comparisons.

### 6.3 The baseline problem

McGreivy and Hakim found that 60 of 76 surveyed fluid-related PDE papers that
claimed improvement over a standard numerical method used a weak baseline. That
finding supplies the program’s methodological prior: every speed or accuracy
claim must name what it beat, at what error, over what rollout, and whether the
comparison included data-generation and training cost.

## 7. Cosmos versioning and the current candidate

### 7.1 The similarly named old tokenizers

The following are different official NVIDIA repositories:

- `nvidia/Cosmos-0.1-Tokenizer-DV8x16x16` — older v0.1 discrete video tokenizer;
- `nvidia/Cosmos-1.0-Tokenizer-DV8x16x16` — later Cosmos 1 tokenizer used by the
  public PhysiX instructions.

The old tokenizer repository is real; it was simply not the later checkpoint
under discussion. NVIDIA now labels the Tokenizer1 collection archived and
points toward Cosmos 3.

### 7.2 Cosmos3-Edge

`nvidia/Cosmos3-Edge` is the current candidate worth investigating first if the
objective is contemporary transfer rather than exact PhysiX reproduction.

Verified properties from the current model card and repository:

- 4B parameters for the overall Edge model;
- BF16 weights under OpenMDW 1.1;
- a Mixture-of-Transformers design with autoregressive and diffusion generation
  mechanisms for different modalities;
- text, image, video, and action capabilities;
- action-conditioned forward dynamics and video-to-action inverse dynamics for
  supported embodiments;
- a Hugging Face repository of approximately 9.18 GB, including approximately
  6.74 GB under `transformer/`, 1.41 GB under `vae/`, and 979 MB under
  `vision_encoder/` at the time checked;
- official Cosmos Framework recipes for supervised post-training. The current
  Edge generator recipe is documented as a full fine-tune. Generic LoRA support
  exists in the framework, but an Edge-generator LoRA recipe for our use case
  was not verified and must not be assumed.

NVIDIA’s training documentation sometimes calls the Edge dense backbone “2B,”
while the Hugging Face model card reports 4B for the complete model. The most
plausible reading is component/backbone count versus the complete multimodal
checkpoint, but the next session should inspect the configs and parameter groups
before budgeting training from either number.

### 7.3 What Edge can do for this program

Native Edge uses RGB/video and embodiment-specific action interfaces. It does
not natively accept arbitrary fields such as pressure, density, three velocity
components, temperature, and dozens of species. A serious numerical adaptation
would need some combination of:

1. an invertible, versioned field normalization and channel representation;
2. input/output channel adapters or a retrained field VAE;
3. reuse of compatible middle VAE and transformer weights;
4. a new conditioning projection for boundary conditions, physical parameters,
   time step, and controls;
5. post-decoding admissibility and conservation enforcement;
6. uncertainty or applicability checks plus classical-solver fallback;
7. comparison with the identical Edge-derived architecture trained from scratch
   and with smaller specialist models.

Packing three fields into RGB can be a pipeline smoke test. A visually pleasing
false-color rollout is not evidence of numerical fidelity and must never be
reported as such.

### 7.4 Why Edge might lose

The non-text generator is diffusion-based. Iterative denoising may be too slow
for a surrogate whose value proposition is speed. A stochastic model may be
valuable for ensembles and uncertainty, but a small deterministic model may win
for routine time stepping. The VAE introduces an error floor before dynamics are
learned. Neither conservation nor units are native. LoRA changes weights; it
does not automatically change the model’s representation or impose physical
constraints.

Cosmos3-Edge is therefore a strong **candidate**, not a presumed solution.

## 8. Memory and BF16 corrections

### 8.1 RTX 3060

The RTX 3060 is an Ampere GPU with compute capability 8.6 and native BF16 Tensor
Core support. BF16 is appropriate for neural-network matrix multiplication on
this card when the installed CUDA/PyTorch stack supports it.

BF16 is not appropriate as the sole precision for authoritative physics:

- use BF16 for most learned weights and activations;
- use FP32 for sensitive losses, normalization, reductions, constraint
  projection, and often optimizer/master state;
- use FP32 or FP64 for the reference solver and final error evaluation.

BF16 and FP16 both occupy two bytes. BF16 has FP32-like exponent range but fewer
mantissa bits than FP16, so normalization and mixed-precision boundaries matter.

The current agent sandbox could not verify the host installation: `nvidia-smi`
could not reach the driver from that execution environment, and its system
Python had no PyTorch module. That is an environment limitation, not evidence
against the 3060 hardware.

### 8.2 Why a 4B Cosmos pipeline can use much more memory than a 4B LLM

A standalone 4B BF16 parameter set is about 8 GB:

\[
4\times10^9\;\text{parameters}\times2\;\text{bytes}\approx8\;\text{GB}.
\]

The parameters are not larger than an LLM’s parameters. A quantized 4B LLM is
smaller still, which is why ordinary 4B language inference can fit on 12 GB.

NVIDIA’s 18.7 GB figure for `Cosmos-1.0-Autoregressive-4B` is an end-to-end video
pipeline measurement under the most extensive documented offloading, not the
weight size of the AR transformer alone. The documented pipeline also involves
the tokenizer/decoder, guardrails, high-resolution video tokens, KV or attention
state, intermediate tensors, and CUDA workspace. NVIDIA reports 31.3 GB with no
offloading and 18.7 GB after offloading the listed components.

Quantization, reduced resolution or sequence length, component-by-component
execution, and custom kernels may change those numbers, but NVIDIA’s model card
only reports BF16 validation. Measure rather than infer feasibility from “4B.”

## 9. What is realistic locally and what belongs on rented GPUs

### 9.1 Local 3060 work

Before downloading a full model:

- define one field dataset and its exact physical meaning;
- build model-independent loading, normalization, inverse normalization, split,
  and metric code;
- establish persistence, linear/low-rank, and small convolutional baselines;
- measure tensor shapes and memory budgets.

After that harness exists, plausible local transfer probes include:

- downloading only the Cosmos3-Edge VAE component and testing reconstruction of
  a carefully defined three-channel field representation;
- training small input/output channel adapters while freezing the VAE core;
- comparing pretrained VAE initialization with the same architecture initialized
  from scratch at fixed data and compute;
- testing BF16 forward/backward kernels with FP32 loss and metrics.

“Plausible” is not “guaranteed to fit.” Batch size, frame count, resolution,
gradient checkpointing, and optimizer state must be measured.

### 9.2 Cloud work

Renting GPU clusters is reasonable only after a local gate establishes that the
representation transfers and after the full benchmark is frozen. Candidate
cloud work includes Edge post-training, multi-seed matched scratch controls, and
long-rollout evaluation. The budget must include the scratch control and
specialist baselines; paying only for the favored model would invalidate the
experiment.

## 10. Custom CUDA C++ and Rust

Building custom infrastructure remains a possible long-term project, but it is
not the first way to test whether the weights are useful.

- **Inference:** plausible after a model and representation win. Relevant pieces
  include safetensors loading, normalization, RoPE/GQA where applicable,
  attention, MLPs, VAE convolutions, diffusion sampling, memory planning, and
  component offload.
- **Training:** do not start by rebuilding PyTorch autograd, distributed
  checkpointing, optimizer sharding, mixed-precision training, and gradient
  checkpointing. Use the reference PyTorch stack for the scientific experiment.
- **Rust:** useful for orchestration, serving, dataset manifests, reproducible
  runs, and process boundaries; CUDA/C++ remains the kernel layer.

A custom CUDA engine is justified by a measured deployment bottleneck after the
model proves useful, not by the existence of open weights.

## 11. Current two-track ordering

### Track A: engineering-first chemistry surrogate

1. Reproduce or adapt a DFODE-Kit workflow on a named chemical mechanism.
2. Generate truth with a named, tuned stiff integrator.
3. Compare a small supervised flow-map model with CVODE/Radau and the practical
   tabulation/manifold incumbent.
4. Enforce admissibility and fall back to the classical solver.
5. Report isolated chemistry and end-to-end CFD speed at matched error.

This track can succeed even if all foundation-model transfer fails.

### Track B: research-first transferred field surrogate

1. Select a concrete field problem and data source.
2. Freeze the representation, units, splits, metrics, horizon, and baselines.
3. Test whether a current pretrained VAE/tokenizer transfers at fixed
   compression against its scratch twin.
4. If and only if that gate passes, adapt a dynamics backbone. Cosmos3-Edge is a
   current candidate; Cosmos-1.0/PhysiX is a historical comparison.
5. Compare transfer, scratch, FNO/TFNO, convolutional, and persistence baselines.
6. Measure long-rollout physical error and solver-relative wall time.

The tracks are related by methodology, not by architecture. Do not force Cosmos
into Track A merely to preserve a unified narrative.

## 12. Decisions that are deliberately not made

- Which physical field dataset should be first.
- Whether the target is combustion instability, general fluid dynamics, or a
  smaller benchmark from The Well.
- Whether the field representation should be continuous or discrete.
- Whether Cosmos3-Edge’s VAE, an updated standalone tokenizer, or another video
  model is the best transferable representation.
- Whether diffusion is acceptable for the required latency.
- Whether LoRA is sufficient or field adaptation needs partial/full fine-tuning.
- Whether uncertainty generation is more valuable than deterministic stepping.
- Whether any full-field surrogate has a business or engineering use case that
  repays data-generation and training cost.

These unresolved choices are why this work is not yet at user-story level.

## 13. Existing local artifacts

- Working thesis: `documents/PHYSICS-ML-BET.md`
- Mathematical chemistry note: `LaTeXandpdfs/SourceTermSurrogate.tex`
- Provisional experimental sketches: `documents/stories/12-*`, `13-*`, `14-*`
- Adversarial roles:
  - `.claude/agents/surrogate-advocate.md`
  - `.claude/agents/transfer-advocate.md`
- Propulsion numerical work: `Cosmos/Source/Numerical/`, `CUDACFD/`,
  `CombustionInstability/`, and `cantera_stuff/`
- Candidate CFD examples: `Data/TurbulentCFDExampleCases/`
- Numerical Recipes agent-context archive:
  `/home/propdev/.openclaw/workspace/Data/Exports/NumericalRecipes3e-AgentContext-2026-08-30.zip`

Numerical Recipes is context for classical numerical methods and strong
baselines. It is not evidence for or against learned surrogates.

## 14. The standard for the next session

The next session should narrow the problem, not expand the model shopping list.
Its best result would be a defensible answer to:

> What is the smallest, cheapest experiment that distinguishes useful pretrained
> physical representation from a visually compelling but numerically irrelevant
> world model?

No model download, cloud rental, or PDD application is authorized by this
handoff. Those become sensible only after the next session fixes the candidate,
dataset, baselines, metrics, and stopping rule.
