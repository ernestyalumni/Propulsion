# The physics-ML bet

A long-horizon track. This document holds the thesis, the evidence for and
against it, and the conditions under which we abandon it. Testable pieces live
in `documents/stories/` (09–12).

## The bet, as stated

Exotic physics-specific architectures — PINNs, Fourier Neural Operators, and
their descendants — are the wrong instrument for real engineering physics.
Better: take an open-weights LLM that already encodes something useful from
pretraining and fine-tune it, rather than train a specialist from scratch.

## What holds up

**The PINN critique is well founded, and specifically so.** The failure observed
applying PINNs to combustion chemical kinetics — reaction rates spanning orders
of magnitude, training diverging — is a known, named, structural problem, not bad
luck. A stiff system carries eigenvalues separated by many decades; the residual
loss is then dominated by the fastest mode, gradients through the slow modes are
comparatively invisible, and the composite loss is ill-conditioned in a way no
learning-rate schedule fixes. It is the same stiffness that forces implicit
integrators on the classical side, where it is met with Jacobians and A-stability
rather than with first-order optimisation.

**The toy-model complaint is fair.** Published PINN and neural-operator results
cluster on Burgers, Allen–Cahn, lid-driven cavity, and low-Reynolds
incompressible flow — problems chosen because they are analytically tractable.
That is a real selection effect worth naming.

## What does not hold up — the two cited repos

Both were read end to end. Neither supports the thesis, and one argues against it.

**`jsjung00/locomotion-language-model`** is a Decision Transformer on HalfCheetah
with a frozen Pythia-410M trunk. Two things matter:

- **The pretrained knowledge is deliberately discarded.** The token embedding
  matrix and the LM head are both replaced with `nn.Identity()`. Roughly 51 M
  parameters — the single largest carrier of learned lexical and semantic
  content — are thrown away, and continuous state vectors are fed straight into
  `inputs_embeds`. No tokenizer is loaded; no text ever enters the model. What
  remains is 24 frozen blocks used as a fixed feature mixer.
- **Its own benchmark contradicts the thesis.** Fine-tuned Pythia-410M scores
  74.9 ± 108.4 at target return 1200. A GPT-2 trained from scratch with 700 K
  parameters scores 409.0 ± 379.8 on the same task — about 5× better at 0.17 % of
  the parameter count. Both sit near 3 % of expert return.

There is also no reinforcement learning in it. Training is supervised regression
under an MSE action loss on a fixed offline dataset; the only reward-shaped
quantity is a return-to-go conditioning token. That is return-conditioned
behaviour cloning.

**`ninjaa/web-mujoco-gym`** contains no open weights and no fine-tuning of any
kind. Claude is called to author reward functions and suggest hyperparameters.
The RL beneath it is inert in two independent ways: the update signal is a mean
of z-scored returns, which is identically zero by construction (measured at
~1e-16, giving per-weight updates near 1e-19), and the policy's actions never
reach the simulator — `nextAction` is written in one place and read in none,
while the physics loop runs on uniform random torques. The repo's own
documentation calls this "Faux RL." A `ppo-take-1` branch contains genuine PPO in
TensorFlow.js, unmerged, whose own summary reports no learning progress.

None of this reflects on the people who wrote them. They are two-day hackathon
projects, and both contain one genuinely good artifact — a clean Decision
Transformer port, and a working Docker→Emscripten→MuJoCo-WASM→Web-Worker
pipeline. They are simply not evidence for the claim being made of them.

## The defensible version of the bet

> **An LLM is a strong *author* of physics code, objectives, and experiment
> configurations, and an unproven *substrate* for a physics surrogate.**

The second repo is real evidence for the first half: given a documented state
schema, a model writes and iteratively refines reward and cost functions — the
Eureka pattern. That transfers directly to GNC cost weights, combustion-stability
figures of merit, trajectory terminal costs, and HIL pass/fail criteria, and it
needs no fine-tuning at all.

The first repo is real evidence for the second half, and it points the wrong way
for us.

This narrower claim is the one to build on, because it survives a reader who
opens the repositories.

## The gap neither repo touches

In both, the network is a **policy** — state to action — and MuJoCo remains the
ground truth carrying state forward. "Fine-tune an LLM for physics simulation"
means the opposite: making the network the map

    (state_t, parameters, boundary conditions) -> state_{t+1}

subject to conservation of mass, momentum and energy; to stability under
autoregressive rollout over 10^3–10^6 steps; and to calibrated error bars. That
is where PINNs and FNOs actually live, and neither cited repo contains a
surrogate model or a comparison against one. Two further gaps: neither has any
notion of units or dimensional consistency, and neither has an
autoregressive-rollout stability story, because the simulator carries the state.

## What the literature actually says (searched 2026-08-30)

**"Large Physics Models" is a real term, and it means the authoring half.**
`arXiv:2501.05382`, published in *European Physical Journal C*, defines LPMs as
LLM-based systems tailored to physics *research* — symbolic manipulation,
literature synthesis, experimental data analysis — not numerical surrogates. The
term that exists names the bet we are calling defensible.

**Physics foundation models for simulation do now exist**, and they are new:
PhysiX (`arXiv:2506.17774`, 4.5 B parameters, discrete tokenizer, autoregressive
next-token over physical states) and the General Physics Transformer
(`arXiv:2509.13805`, 1.8 TB of simulation data, reports inferring governing
dynamics from context and plausible zero-shot transfer to unseen systems). Also
PDE-FM and SPUS, benchmarked on *The Well*. **Read the comparison carefully:**
GPhyT's headline "7x" is against *specialized neural architectures*, not against
a classical solver. Data scarcity is the stated bottleneck — the largest physics
simulation datasets hold tens of thousands of samples against internet-scale text.

**The field has a documented, quantified overoptimism problem.** McGreivy &
Hakim, *Nature Machine Intelligence* 6:1256-1269 (2024), `arXiv:2407.07218`:
of articles claiming to beat a standard numerical method on a fluid-related PDE,
**79% (60 of 76) compared against a weak baseline**. They also document outcome
reporting bias and publication bias. This is the strongest single piece of
support for the sceptical prior in this document, and it is the methodological
template for story 09.

**For stiff chemical kinetics specifically, something does work — and it is not
a PINN.** The distinction that matters:

- A **PINN** represents the solution field and trains on the PDE residual. On a
  stiff system this fails structurally, which is the failure already recorded
  above.
- A **source-term surrogate** replaces one substep inside a classical solver:
  the chemistry ODE integration. The CFD solver, the transport, and the
  conservation structure all stay. It is trained supervised on data generated by
  CVODE. This is the modern descendant of in-situ adaptive tabulation and
  flamelet manifolds, and it ships.

Concrete results:

- **DFODE-Kit**, *Computer Physics Communications* (2025): O(10^2) acceleration
  on isolated chemistry evaluation against CVODE, and up to **20x end-to-end CFD
  speedup**. Open source, integrates with DeepFlame 2 and adaptable to OpenFOAM.
  Covers hydrogen, hydrocarbons and ammonia; premixed flames, detonation, ignition.
- `arXiv:2507.08277` (ammonia/natural gas) reports the fix for exactly the
  failure that motivated this bet: **scale separation for targets spanning
  multiple orders of magnitude**, plus physics-aware augmentation for the
  sampling imbalance near steep flame-front gradients.
- Physics-constrained neural ODEs, *Combustion Theory and Modelling* 29(3)
  (2025), put mass conservation in the loss; other work enforces mass, energy
  and element balance as hard constraints. ChemKANs (*PCCP* 2025) is a further
  variant.

The lesson is that the winning formulation is modest. It does not ask a network
to learn physics. It asks a network to interpolate one expensive function
evaluation, inside a framework that still enforces the physics.

## On the Witten analogy

Edward Witten read history and then journalism before graduate physics, and it is
a fair prompt for the hypothesis. But it cuts both ways, and the honest reading
supports a different conclusion than the one usually drawn from it: Witten did
not arrive already knowing physics because he had read widely. He did years of
concentrated graduate training. The analogy argues for **extensive fine-tuning on
domain data**, not for latent physics competence in a text-pretrained model. Read
that way it supports the programme; read the other way it is the thing that needs
proving.

## How this bet gets settled

The bet is only worth holding if it can lose. Story 09 makes the adversarial
check standing rather than occasional. The falsifiers, fixed in advance:

- **The bet loses** if, on a stiff benchmark with a documented stiffness ratio, a
  fine-tuned open-weights model fails to beat both a from-scratch transformer of
  comparable parameter count and a competently tuned neural operator.
- **The bet also loses** if it can only be made to win by discarding the
  pretrained embeddings, since that removes the thing under test.
- **The PINN rejection loses** if the contrarian track produces a stiff-kinetics
  PINN result that a classical implicit integrator does not dominate on accuracy
  per unit compute.

## Sequencing

- **Track 1 — the adversary** (story 09). Runs from the start, permanently.
- **Track 2 — architecture** (story 10). Narrower than first scoped: the
  attention ladder is already built in CuLLM, scalar through WMMA to CuTe, and
  Episode 1 of the attention series is recorded. What is missing is the modern
  stack — mixture-of-experts routing, grouped-query attention, RoPE, and modern
  normalisation placement.
- **Track 3 — reinforcement learning** (story 11). From zero, and the two repos
  are a warning about what "has RL" can mean.
- **Track 4 — the experiment** (story 12). The actual fine-tune, with the
  baseline repo 1 accidentally ran and lost.
