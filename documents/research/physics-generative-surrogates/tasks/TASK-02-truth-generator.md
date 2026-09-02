# TASK-02 — Truth generator, regime splits, label validation

**Goal.** An HDF5 dataset of chemistry flow-map samples
(T, p, Y, Δt) → (T', Y') from cold-start CVODE, with splits that hold out whole
regimes, and a validator proving every label is physically admissible.
**Why.** This is the training and test data for E1 and the object every
baseline is scored on.
**Depends on.** TASK-01; decision 1 (default: CH4/O2 with ffcm1; run h2o2 first as a smoke test).
**Effort.** 1 day of coding; generation runs in minutes to an hour.

## Envelope (defaults from CANDIDATE-EXPERIMENTS §6.1; overridable by a YAML config)
- oxidizer O2 (no N2); p 100–350 bar; φ 0.5–2.0; T 700–4000 K; Δt ∈ {1e-8, 1e-7, 1e-6} s.
- h2o2 smoke test: p 1–10 atm, T0 1000–1500 K, φ 0.5–2, Δt ∈ {1e-7, 1e-6}, with air.

## Sampling (manifold-aware; DFODE-Kit is the reference for the idea)
1. Trajectory sampling: run 0-D constant-pressure ignition from a Latin-hypercube
   of (T0, p, φ) initial conditions; take states log-spaced in time from 1e-3 τ_ign to 20 τ_ign.
2. Augmentation transverse to the manifold: for a fraction of samples, perturb
   log(Y) by N(0, σ²) with σ ∈ {0.1, 0.3}, project back onto element conservation
   (the (I − E⁺E) projector from `LaTeXandpdfs/SourceTermSurrogate.tex`), renormalize, keep T.
3. For every sampled state and every Δt: cold-start CVODE (`gas.TPY=…; r.syncState();
   net.initial_time=0; net.reinitialize(); net.advance(Δt)`) at rtol 1e-8, atol 1e-15.

## Splits
Hold out entire bands, never neighbours: test = φ ∈ [0.5, 0.6] ∪ [1.8, 2.0] and
p ∈ [300, 350] bar; validation = a random 10% of the remaining trajectories by
trajectory id; train = the rest. Write `splits.json` with the rule and the ids.

## Validator (must run and pass before the file is accepted)
For every label: Y' ≥ 0; |ΣY' − 1| < 1e-12; ‖E(Y' − Y)‖∞ < 1e-10 (element
conservation); |h(T', Y') − h(T, Y)| / |h| < 1e-8 (enthalpy at constant p);
T' finite. Report the count of failures and drop nothing silently.

## Outputs
- `Surrogates/chem/generate_truth.py`, `Surrogates/chem/validate_truth.py`, `Surrogates/chem/config/*.yaml`
- `/media/propdev/Expansion/openclaw/.openclaw/workspace/Data/Public/datasets/chem-flowmap/<mech>-<date>/{data.h5,splits.json,MANIFEST.md}`
  with attributes: mechanism file hash, Cantera version, tolerances, envelope, seed, sample counts.
- `Surrogates/results/02-REPORT.md` with sample counts per split, generation wall time, validator output.

## Done when
- h2o2 smoke set (≥ 2e5 samples) and the decision-1 set (≥ 1e6 samples) exist, validate with zero failures, and the split rule is in the manifest.

## Do not
- Do not sample the test bands during training-set generation. Do not lower the truth tolerance to go faster.
