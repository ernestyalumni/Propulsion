# Turns, *An Introduction to Combustion*, 3rd ed. (McGraw-Hill, 2012) — reading roadmap

**Written:** 2026-09-29, for the simulation plan in [`../SIMULATION-READING-PLAN.md`](../SIMULATION-READING-PLAN.md).
**Corpus:** `<CORPUS_ROOT>/Public/books/EngineeringPhysics/Turns-IntroductionToCombustion-3e/`
(reading copy `parsed/book-reconciled.md`). Page numbers are printed folios; PDF = printed + 19 for PDF pages 20–753.

**Why this book:** It is the combustion primer: thermochemistry, kinetics, reactor models, droplets. It has no solid-propellant chapter. Chapter 14 is carbon and coal, and aluminium is not treated. For the solid booster it supplies two things: the adiabatic flame temperature and equilibrium products (Ch. 2), which give T₀, 𝓜 and γ, and so c*; and the open control-volume reactor balances (Ch. 6), which are the chamber free-volume and plenum equations with the reaction switched off.

**Read it with:** Sutton ch. 5, Humble ch. 4 and H&P §2.4, §12.3 for thermochemistry. Read it before Williams. The track's data is [`../SOLID-MOTOR-TRACK.json`](../SOLID-MOTOR-TRACK.json).

## Chapter ranking

Ranked for the solid-booster simulation first, then the liquid engine. `read-only` chapters inform the models but do not become code.

| Rank | Chapter | Printed page | Why it earns its place | Becomes | Notes |
|---|---|---:|---|---|---|
| 1 | 2 Combustion and Thermochemistry | 12 | adiabatic flame temperature; equilibrium products by Gibbs minimization; the chamber state (T0, molar mass, gamma) that sets c* | propulsion::thermochemistry | for a solid, T_ad at chamber pressure with condensed Al2O3 is the T0 the ballistics uses |
| 2 | 6 Coupling Chemical and Thermal Analyses of Reacting Systems | 183 | constant-volume and well-stirred reactors: open control-volume mass and energy balances, Eqs. (6.28), (6.34) | propulsion::solid_ballistics | with no reaction this is exactly the gas-generator plenum and the SRM free volume |
| 3 | 4 Chemical Kinetics | 107 | elementary rates, steady-state and partial-equilibrium approximations, time scales | propulsion::kinetics | liquid-track Phase 3 |
| 4 | 5 Some Important Chemical Mechanisms | 149 | H2-O2, CH4 mechanisms and reduced mechanisms | propulsion::kinetics | liquid-track Phase 3 |
| 5 | 10 Droplet Evaporation and Burning | 366 | d-squared law, droplet burning, the liquid-rocket application | read-only | liquid-track Phase 7; first model for aluminium-droplet burning time in a solid-motor port |
| 6 | 3 Introduction to Mass Transfer | 79 | Fick's law, Stefan problem | read-only | prerequisite for ch. 10 and 14 |
| 7 | 14 Burning of Solids | 527 | heterogeneous reactions; carbon one-film and two-film models | read-only | not propellants: an analog for a burning metal particle only |
| 8 | 7 Simplified Conservation Equations for Reacting Flows | 220 | reacting-flow conservation equations, conserved scalar | read-only | prerequisite for Williams |
| 9 | 8 Laminar Premixed Flames | 258 | flame speed and structure | read-only | background for Williams 7.5 |
| 10 | A Selected Thermodynamic Properties of Gases Comprising C-H-O-N System | 686 | property tables | propellants::properties |  |
| 11 | E Generalized Newton's Method for the Solution of Nonlinear Equations | 710 | Newton iteration for the equilibrium solver | propulsion::thermochemistry |  |
| 12 | F Computer Codes for Equilibrium Products of Hydrocarbon-Air Combustion | 713 | equilibrium codes | read-only |  |
| 13 | 16 Detonations | 616 | detonations | read-only |  |
| 14 | 9 Laminar Diffusion Flames | 311 | jet flames | read-only |  |
| 15 | 11 Introduction to Turbulent Flows | 427 | turbulence | read-only |  |
| 16 | 12 Turbulent Premixed Flames | 453 | turbulent premixed | read-only |  |
| 17 | 13 Turbulent Nonpremixed Flames | 486 | turbulent nonpremixed | read-only |  |
| 18 | 15 Emissions | 556 | pollutant formation | read-only |  |
| 19 | 17 Fuels | 638 | fuel properties | read-only |  |
| 20 | 1 Introduction | 1 | landscape | read-only |  |
| 21 | B Fuel Properties | 700 | fuel property tables | read-only |  |
| 22 | C Selected Properties of Air, Nitrogen, and Oxygen | 704 | property tables | read-only |  |
| 23 | D Binary Diffusion Coefficients and Methodology for their Estimation | 707 | diffusion coefficients | read-only |  |
