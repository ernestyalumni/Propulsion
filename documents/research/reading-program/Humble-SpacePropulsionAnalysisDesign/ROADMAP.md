# Humble, Henry and Larson, *Space Propulsion Analysis and Design* (McGraw-Hill, 1995) — reading roadmap

**Written:** 2026-09-24, for the simulation plan in [`../SIMULATION-READING-PLAN.md`](../SIMULATION-READING-PLAN.md).
**Corpus:** `<CORPUS_ROOT>/Public/books/EngineeringPhysics/Humble-SpacePropulsionAnalysisDesign/`
(reading copy `parsed/book-reconciled.md`). Page numbers are printed folios; PDF = printed + 20 for
PDF pages 21–753.

**Why this book:** it is the system designer's view. It sizes a whole propulsion system from mission
requirements, and it closes each major chapter with a worked case study. Chapter 6 is the clearest
solid-motor *performance prediction* treatment in the library: §6.5.1 lumped-parameter ballistics and
§6.5.2 ballistics with spatial pressure variation.

**Read it with:** Sutton ch. 3–6 and 12–15, H&P ch. 3 and §12.6–12.7, Huzel ch. 4–6. The row-by-row
overlap is on the reading room's roadmap page and in [`../FOUR-BOOK-OVERLAP.json`](../FOUR-BOOK-OVERLAP.json).

## Chapter ranking

Ranked for the liquid-engine and solid-booster simulations. `read-only` chapters inform the models but do not become code.

| Rank | Chapter | Printed page | Why it earns its place | Becomes | Notes |
|---|---|---:|---|---|---|
| 1 | 6 Solid Rocket Motors | 295 | solid-motor design process, sizing of case, igniter, insulation, nozzle and TVC; propellant and burning rate; performance prediction by lumped-parameter and spatial-pressure ballistics; a full case study | propulsion::solid_ballistics | pairs Sutton ch. 12-15 and H&P 12.6-12.7; case study 6.6 is the SRM golden vector |
| 2 | 5 Liquid Rocket Propulsion Systems | 179 | preliminary design decisions, pressure budget, tank sizing, thrust chamber, feed system, turbomachinery, pressurization; case study | propulsion::cycle | pairs Sutton ch. 6 and Huzel ch. 4-6; case study 5.5 is an LRE golden vector |
| 3 | 4 Thermochemistry | 149 | heats of formation, equilibrium-constant and free-energy-minimization methods, flame temperature, kinetics overview | propulsion::thermochemistry | pairs Sutton ch. 5, H&P 2.4, Turns ch. 2 |
| 4 | 3 Thermodynamics of Fluid Flow | 77 | control-volume first law, isentropic flow, thrust equation, C_F and c*, heat addition, conduction, convection, radiation; cold-gas thruster example | propulsion::nozzle | pairs Sutton ch. 3 and H&P ch. 3 |
| 5 | 2 Mission Analysis | 31 | orbits, perturbations, maneuvers, Earth-to-orbit velocity budget, staging, launch-vehicle steering, flight-simulation programs | flight::ascent | pairs Sutton ch. 4 |
| 6 | C Launch Vehicles and Staging (Appendix C) | 715 | launch-vehicle data and staging | flight::staging |  |
| 7 | B Thermochemical Data for Selected Propellants (Appendix B) | 695 | tabulated propellant thermochemistry | propellants::properties |  |
| 8 | 10 Mission Design Case Study | 599 | end-to-end propulsion selection for a mission | read-only |  |
| 9 | 7 Hybrid Rocket Propulsion Systems | 365 | hybrid regression and design | read-only |  |
| 10 | 1 Introduction to Space Propulsion | 1 | landscape | read-only |  |
| 11 | 9 Electric Rocket Propulsion Systems | 509 | electric propulsion | read-only |  |
| 12 | 8 Nuclear Rocket Propulsion Systems | 443 | nuclear thermal | read-only |  |
| 13 | 11 Advanced Propulsion Systems | 631 | advanced concepts | read-only |  |
