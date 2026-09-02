# TASK-12 — Model-independent field harness and trivial baselines

**Goal.** Story 13's first milestone: loader, normalization and its inverse,
regime-held-out splits, physical metrics, and trivial baselines, all verified
before any learned model runs.
**Depends on.** TASK-11.
**Effort.** 2 days.

## Build
- `Surrogates/fields/data.py`: reads the BLASTNet subset into tensors
  [trajectory, time, channel, y, x] (2-D slices of 3-D data are acceptable; record
  the slicing rule). Per-channel normalization fit on train only; exact inverse;
  units retained in metadata.
- Splits: hold out whole trajectories or whole parameter values; never frames from a seen trajectory.
- `Surrogates/fields/metrics.py`: per-field VRMSE and RMSE in physical units after
  inverse normalization; radial energy spectrum error; conserved integrals
  (total mass, momentum; energy if fields allow); boundary-condition violation;
  rollout divergence time (first step where VRMSE exceeds a threshold).
- Baselines: persistence (u_{t+1} = u_t), linear extrapolation, and a per-pixel
  linear regression from the last k frames. Report all metrics on validation and test.
- Tests: normalization round-trip < 1e-6; metrics on a known synthetic field.

## Outputs
- `Surrogates/fields/{data.py,metrics.py,baselines_trivial.py,tests/}`, `Surrogates/results/12-trivial-baselines.{json,md}`, `Surrogates/results/12-REPORT.md`.

## Done when
- Tests pass, and the trivial-baseline table exists for 1-step and 20-step rollouts on the held-out split.
