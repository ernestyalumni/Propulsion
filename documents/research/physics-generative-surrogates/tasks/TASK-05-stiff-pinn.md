# TASK-05 — Stiff-PINN reproduction and the H2/O2 attempt

**Goal.** The strongest published PINN for stiff kinetics, run with its authors'
code and hyperparameters, first on their own problems and then on our H2/O2
ignition case; plus the vanilla PINN failure with its loss conditioning measured.
**Why.** Story 09: a claim that PINNs cannot handle stiffness must be tested
against the best version of a PINN, not a strawman.
**Depends on.** TASK-02 (h2o2 set). Needs the 3060.
**Effort.** 1–2 days.

## Steps
1. Clone https://github.com/DENG-MIT/Stiff-PINN into `Surrogates/baselines/external/Stiff-PINN`
   (record the commit hash). Create a **separate** uv venv there if its dependencies
   conflict with ours; never modify the Surrogates venv to suit it.
2. Reproduce the authors' ROBER and POLLU results with their scripts and
   hyperparameters unchanged. Save loss curves and final errors. If reproduction
   fails, that is the finding; record exact errors.
3. Vanilla PINN on h2o2 0-D ignition (T0 1200 K, 1 atm, φ 1): the residual loss
   of `SourceTermSurrogate.tex` eq. (pinnloss) over t ∈ [0, 3 τ_ign] with the
   network representing (T, Y)(t). Train for a fixed budget. Report the loss
   curve, the final error against CVODE, and an estimate of the Gauss–Newton
   Hessian condition number from the Jacobian spectrum in TASK-03.
4. Stiff-PINN on the same case: derive the QSSA-reduced system for H2/O2 the
   way the authors did for their problems (QSS species: H, O, OH, HO2, H2O2 as a
   first choice; record what you chose and why). Train with their settings.
   Report accuracy versus CVODE and wall time to reach it.
5. Accuracy per unit compute for both PINNs against cold-start CVODE from TASK-03.

## Outputs
- `Surrogates/baselines/pinn/` (our scripts), `Surrogates/results/05-stiff-pinn.{json,md}`, `Surrogates/results/05-REPORT.md`, loss-curve PNGs.

## Done when
- Rows exist for ROBER, POLLU, vanilla-PINN-H2, Stiff-PINN-H2 with error vs CVODE, training wall time, inference cost, and the retained negative results.

## Do not
- Do not tune the PINNs beyond the authors' published settings. Do not weaken the CVODE reference.
