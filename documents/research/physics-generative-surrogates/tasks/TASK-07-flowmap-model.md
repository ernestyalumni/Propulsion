# TASK-07 — Flow-map surrogate: transforms, projection, training, a priori metrics

**Goal.** Train the E1 network on the TASK-02 data with hard physical
constraints, and report a priori held-out accuracy per species in physical units.
**Depends on.** TASK-02. Needs the 3060.
**Effort.** 2 days.

## Model (follow `LaTeXandpdfs/SourceTermSurrogate.tex` §4 exactly)
- Inputs: T/1000, log10(p/1e5), ψ_k = log(Y_k + ε) standardized per species, log10(Δt).
- Output: raw ΔY̌ in transformed space; map to ΔY; project ΔY ← (I − E⁺E) ΔY with
  E the element matrix plus the all-ones row (total mass); Y' = Y + ΔY; clip ≥ 0; renormalize.
- Temperature: **not predicted**. Solve h(T', Y') = h(T, Y) by Newton with Cantera's
  thermo (2–3 iterations from T). Batched.
- Network: residual MLP, width 256–512, 3–6 blocks, SiLU; 0.1–2M parameters.
- Loss: MSE in standardized ψ space on ΔY; FP32 loss and metrics; BF16 autocast allowed for matmuls.
- Optimizer AdamW, cosine schedule, fixed step budget; 3 seeds.

## Evaluation (a priori only; a posteriori is TASK-08)
- Per-species log-relative error and absolute error, T error after the enthalpy
  solve, ‖EΔY‖∞ (must be ≤ 1e-12 by construction; verify), on validation and on the
  held-out bands separately.
- Wall time per sample batched on the 3060 (batch 10^4, 10^5), FP32 and BF16.

## Outputs
- `Surrogates/chem/model.py`, `Surrogates/chem/train.py`, `Surrogates/chem/evaluate.py`,
  checkpoints under `Surrogates/results/07-ckpt/`, `Surrogates/results/07-apriori.{json,md}`, `Surrogates/results/07-REPORT.md`.

## Done when
- Three seeds trained on h2o2 and on the decision-1 set; the metric table has one row per (split, seed) with per-species errors, and the projection residual is at machine precision.

## Do not
- Do not look at the test bands while choosing width, depth, or schedule. Do not add a temperature output.
