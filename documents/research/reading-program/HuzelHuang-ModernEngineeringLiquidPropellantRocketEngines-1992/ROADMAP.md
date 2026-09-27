# Huzel and Huang, *Modern Engineering for Design of Liquid-Propellant Rocket Engines* (AIAA, 1992) — reading roadmap

**Written:** 2026-09-24, for the simulation plan in [`../SIMULATION-READING-PLAN.md`](../SIMULATION-READING-PLAN.md).
**Corpus:** `<CORPUS_ROOT>/Public/books/EngineeringPhysics/HuzelHuang-ModernEngineeringLiquidPropellantRocketEngines-1992/`
(reading copy `master/book.md`). Page numbers are printed folios. The downloaded PDF has
duplicate renders and six unobserved folios (66, 154, 284, 344, 404, 414), so PDF pages come from
`reference/downloaded-page-map.json`, not a constant offset.

**Why this book:** it is the liquid-engine design manual. Sutton gives the handbook view and Hill and
Peterson the derivations. Huzel and Huang gives the design procedures, with worked sample calculations
on four reference engines. Its §10.2 (pp. 346–350) is the closest thing in the library to a spec for
a transient engine simulation.

**Read it with:** Sutton ch. 6, 8, 10, 11 and H&P ch. 4, 13. The row-by-row overlap is on the
reading room's roadmap page and in [`../FOUR-BOOK-OVERLAP.json`](../FOUR-BOOK-OVERLAP.json).

## Chapter ranking

Ranked for the liquid-engine and solid-booster simulations. `read-only` chapters inform the models but do not become code.

| Rank | Chapter | Printed page | Why it earns its place | Becomes | Notes |
|---|---|---:|---|---|---|
| 1 | 4 Design of Thrust Chambers and Other Combustion Devices | 67 | chamber layout and L*, nozzle shape, gas-side and coolant-side heat transfer, regenerative/film/ablative cooling, injector orifice sizing, gas generators, igniters, instability | propulsion::thrust_chamber, propulsion::cooling | read H&P ch. 4 before 4.4; pairs Sutton ch. 8 |
| 2 | 10 Engine Systems Design Integration | 345 | 10.2 is the engine-system dynamic model: equations, start/shutdown transients, engine-vehicle interaction, low-frequency instability; 10.3-10.4 calibration and influence coefficients | propulsion::engine_dynamics, propulsion::calibration | pairs Sutton ch. 11; the closest thing in the library to a transient engine-simulation spec |
| 3 | 6 Design of Turbopump Propellant Feed Systems | 155 | pump and turbine design parameters, inducers and NPSH, centrifugal and axial pumps, turbine types, rotordynamics | propulsion::turbopump | pairs Sutton ch. 10 and H&P ch. 13 |
| 4 | 3 Introduction to Sample Calculations | 53 | four sample stage engines (A-1 to A-4) with cycles and start/cutoff sequences; the book's worked calculations build on them | golden vectors for propulsion::cycle | use as reference cases, not as a module |
| 5 | 7 Design of Rocket-Engine Control and Condition-Monitoring Systems | 219 | thrust and mixture-ratio control, control laws, valves and regulators, instrumentation, failure detection, post-flight data analysis | propulsion::engine_control | pairs Sutton ch. 11 |
| 6 | 5 Design of Gas-Pressurized Propellant Feed Systems | 135 | pressurant mass calculations, stored-gas, evaporation and chemical-reaction pressurization | propulsion::pressurization | pairs Sutton 6.4-6.5 |
| 7 | 1 Introduction to Liquid-Propellant Rocket Engines | 1 | thrust, chamber and nozzle flow, Is, c*, Cf, correction factors, propellant property tables | propulsion::performance (cross-check) | pairs Sutton ch. 2-3 |
| 8 | 2 Engine Requirements and Preliminary Design Analyses | 23 | design parameters, mission requirements, preliminary design optimization | read-only |  |
| 9 | 8 Design of Propellant Tanks | 285 | tank structure, cryogenic insulation, zero-g expulsion | read-only |  |
| 10 | 9 Design of Interconnecting Components and Mounts | 305 | ducts and pressure drop, pump-inlet line vibration, bellows, gimbal mounts | read-only | 9.1 only if modelling line dynamics or pogo |
| 11 | 11 Design of Liquid-Propellant Space Engines | 373 | spacecraft main propulsion and RCS | read-only |  |
| 12 | A Weight, Reliability, Materials (Appendices A-C) | 389 | weight estimation, reliability, materials | read-only |  |
