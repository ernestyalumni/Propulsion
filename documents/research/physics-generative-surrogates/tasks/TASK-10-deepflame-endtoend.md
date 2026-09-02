# TASK-10 — Surrogate inside DeepFlame, end-to-end wall time

**Goal.** The only number that supports a "faster simulation" claim: wall time of
a DeepFlame case with the E1 surrogate versus the same case with CVODE, at
matched acceptable error, with the fallback rate.
**Depends on.** TASK-08 GO verdict, TASK-09.
**Effort.** 2 days.

## Steps
1. Export the TASK-07 model to the tensor contract in `DNN-INTERFACE.md`
   (TorchScript or whatever DeepFlame loads); include the transforms, projection,
   and gate inside the exported graph or in the C++ wrapper — record which.
2. Run the TASK-09 case with CVODE and with the surrogate; identical mesh, steps, tolerances.
3. Compare: flame position/speed, peak T, major species, enthalpy drift, mass
   conservation, fallback fraction, and end-to-end wall time; also report
   chemistry-only wall time.
4. Compute the break-even count: (truth generation + training wall time) / (per-run saving).

## Outputs
- `Surrogates/deepflame/export_model.py`, `Surrogates/results/10-endtoend.{json,md}`, `Surrogates/results/10-REPORT.md`.

## Done when
- The end-to-end table has both runs, the error columns, the speedup, and the break-even count, and states the Amdahl ceiling from TASK-09 beside the achieved number.
