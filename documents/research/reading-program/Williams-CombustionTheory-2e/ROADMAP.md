# Williams, *Combustion Theory*, 2nd ed. (Benjamin/Cummings, 1985) — reading roadmap

**Written:** 2026-09-29, for the simulation plan in [`../SIMULATION-READING-PLAN.md`](../SIMULATION-READING-PLAN.md).
**Corpus:** `<CORPUS_ROOT>/Public/books/EngineeringPhysics/Williams-CombustionTheory-2e/`
(reading copy `parsed/book-reconciled.md`). Page numbers are printed folios; PDF = printed + 23 for PDF pages 24–399 (printed 377 is missing from the scan), + 22 for 400–541, + 21 for 542–670.

**Why this book:** It is the combustion theory. For the solid booster, Chapter 7 explains why a propellant burns as r = a pⁿ with n < 1, what sets the temperature sensitivity, how composite (AP) propellants burn, and gives the erosive-burning correlations. Chapter 9 has the solid-motor instabilities: acoustic modes (§9.1), intrinsic oscillations and chuffing (§9.2), and the bulk-mode or L* instability (§9.4).

**Read it with:** Sutton ch. 12 and 14, H&P §12.6–12.8, Humble §6.4.5. Read Turns first for thermochemistry and kinetics. The solid-motor track on the reading room's roadmap page shows which Williams sections go with each simulation step. The track's data is [`../SOLID-MOTOR-TRACK.json`](../SOLID-MOTOR-TRACK.json).

## Chapter ranking

Ranked for the solid-booster simulation first, then the liquid engine. `read-only` chapters inform the models but do not become code.

| Rank | Chapter | Printed page | Why it earns its place | Becomes | Notes |
|---|---|---:|---|---|---|
| 1 | 7 Combustion of Solid Propellants | 229 | homogeneous deflagration (7.1); condensed- vs gas-phase control (7.4, 7.5) — why n < 1; Eq. (7-41) temperature law (p. 250); heterogeneous AP composites (7.7); erosive burning, Eqs. (45)-(46) (7.8) | propulsion::solid_ballistics | read 7.1, 7.5, 7.7, 7.8 first; 7.2-7.4 and 7.6 when a model needs them. Sutton p. 445: a and n are still measured, not computed |
| 2 | 9 Combustion Instabilities | 294 | SRM acoustic modes, admittance, damping incl. particle damping (9.1); intrinsic oscillations of burning solids and chuffing (9.2); bulk-mode / L* instability (9.4) | propulsion::stability | 9.4 first: it is the instability the lumped chamber ODE can show |
| 3 | 4 Reactions in Flows with Negligible Molecular Transport | 92 | ignition delay and the well-stirred reactor (4.1); reacting quasi-1-D nozzle flow, rocket Isp, freezing (4.2); two-phase nozzle flow (4.2.6) | propulsion::nozzle | 4.2.6 is the Al2O3 two-phase loss for the solid nozzle |
| 4 | A Summary of Applicable Results of Thermodynamics and Statistical Mechanics | 521 | equilibrium thermodynamics behind the chamber state | propulsion::thermochemistry | pairs Turns ch. 2, Sutton ch. 5 |
| 5 | B Review of Chemical Kinetics | 554 | mass action, chain reactions, rate theory | propulsion::kinetics | liquid-track Phase 3 |
| 6 | 8 Ignition, Extinction, and Flammability Limits | 265 | minimum ignition energy, heat-loss flames, activation-energy asymptotics | read-only | take with Sutton 14.2 only if the ignition transient becomes a focus |
| 7 | 11 Spray Combustion | 446 | spray combustion theory | read-only | liquid-track Phase 7 |
| 8 | 3 Diffusion Flames and Droplet Burning | 38 | Burke-Schumann, droplet burning | read-only | liquid-track Phase 7; aluminium-droplet analog |
| 9 | 5 Theory of Laminar Flames | 130 | premixed flame structure and speed | read-only | background for 7.5 gas-phase flames |
| 10 | 1 Summary of Relevant Aspects of Fluid Dynamics and Chemical Kinetics | 1 | conservation equations and kinetics summary | read-only |  |
| 11 | 2 Rankine-Hugoniot Relations | 19 | deflagration and detonation jump conditions | read-only |  |
| 12 | 6 Detonation Phenomena | 182 | detonations | read-only |  |
| 13 | 10 Theory of Turbulent Flames | 373 | turbulent flames | read-only |  |
| 14 | 12 Flame Attachment and Flame Spread | 485 | flame attachment and spread | read-only |  |
| 15 | C Continuum Derivation of the Conservation Equations | 604 | conservation-equation derivation | read-only |  |
| 16 | D Molecular Derivation of the Conservation Equations | 618 | kinetic-theory derivation | read-only |  |
| 17 | E Transport Properties | 628 | transport coefficients | read-only |  |
