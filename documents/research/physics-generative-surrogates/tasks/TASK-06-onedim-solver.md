# TASK-06 — 1-D operator-split reaction–diffusion solver

**Goal.** A small unsteady 1-D flame solver with a swappable chemistry substep,
validated against Cantera's steady flame speed, so surrogates can be tested a
posteriori in a coupled setting under our control.
**Depends on.** TASK-01. Independent of the dataset.
**Effort.** 2 days.

## Design
- Variables: ρ, T, Y_k on a uniform grid; constant pressure; low-Mach (no acoustics).
- Strang splitting per step Δt: transport Δt/2 → chemistry Δt → transport Δt/2.
- Transport: explicit or semi-implicit diffusion with mixture-averaged D_k, λ from
  Cantera's `transport_model='mixture-averaged'`; advection by the inflow velocity.
- Chemistry substep interface: `chem.advance(T, p, Y, dt) -> (T', Y')` for a whole
  grid at once (batched). Two implementations at first: cold-start Cantera CVODE
  per cell, and a "pass-through" (no chemistry) for testing transport alone.
- Boundary: inflow at fixed (T, Y, u), outflow zero-gradient.
- Written in NumPy first; a PyTorch version of transport is optional.

## Validation
1. Diffusion-only test against an analytic Gaussian spreading solution (error < 1e-3).
2. Freely propagating premixed flame at 1 atm and 10 atm, h2o2 and gri30_cho
   with air: compare the steady flame speed (from the consumption rate of fuel)
   with Cantera `FreeFlame` for the same mechanism and transport; agree within 3%
   with grid resolution reported. Then attempt 100 bar and report whether the
   grid becomes impractical (flame thickness in µm).
3. Report chemistry share of wall time f for the CVODE substep.

## Outputs
- `Surrogates/onedim/` (solver, tests), `Surrogates/results/06-onedim-validation.{json,md}`, `Surrogates/results/06-REPORT.md`.

## Done when
- Flame speeds match Cantera within 3% at 1 and 10 atm for both mechanisms, and f is reported.
