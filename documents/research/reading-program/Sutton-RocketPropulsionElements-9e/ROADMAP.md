# Sutton and Biblarz, *Rocket Propulsion Elements*, 9th ed. — reading roadmap and coding briefs

**Status:** written 2026-09-02 from the verified table of contents; OCR of the
full text is running. PDF page = printed page + 24 throughout (`INDEX.md` in
the corpus slug has every section and subsection).
**Corpus slug:** `Data/Public/books/EngineeringPhysics/Sutton-RocketPropulsionElements-9e/`
**Edition note:** this is the 2017 Wiley 9th edition (title page verified). A
10th edition exists; when a copy arrives, re-anchor this roadmap's page
numbers and keep the section numbering here as the citation of record until
then.
**Why this book:** it is the propulsion counterpart of Wie: nozzle theory,
flight performance, engine cycles, turbopumps, thrust chambers, controls and
calibration, combustion stability, TVC, and testing. It is the direct
counterpart of the Raptor propulsion-simulation posting's lumped-parameter
thermofluid engine model and of the "post-flight data correlation" line in
the GNC posting (`../../numerical-recipes-rewrite/spacex-signal-2026-09-01.md`).
**Rules that apply:** stories 15 and 16; the seven-step protocol in
`../../numerical-recipes-rewrite/ROADMAP.md` §2.
**Already in the repository (port, do not rewrite blind):** `NozzleTheory.py`,
`Propulsion.py`, `thermo.py`, `Liquiddensity.py`, `LiquidVaporEq.py`,
`cantera_stuff/` (LOX/methane equilibrium, frozen-versus-shifting), `ccdroplet/`,
`CombustionInstability/`, `Surrogates/` (chemistry stiffness measurements).

## 1. Chapter ranking

Tier 0: §2.1–2.3 (26–35) for the vocabulary, and Appendix 3 (749) which is
the list of key equations every property test below is built from.

| Rank | Chapter | The physics that earns its place | Sections (printed page) | Becomes | Language | Depends on |
|---|---|---|---|---|---|---|
| 1 | 3 Nozzle Theory and Thermodynamic Relations (45–98) | The ideal rocket: isentropic expansion, throat condition, thrust coefficient, characteristic velocity, specific impulse, area ratio ↔ pressure ratio, under- and over-expansion and separation, conical and bell divergence loss, real-nozzle correction factors | 3.1 (45), 3.2 (47), 3.3 (51), 3.4 (73), 3.5 (81), 3.6 (91) | `propulsion::nozzle` | Rust (port `NozzleTheory.py`); C++ twin only if the WASM demo wants it | NR 9 (area-ratio inversion) |
| 2 | 2 Definitions and Fundamentals (26–44) | Thrust, exhaust velocity, effective exhaust velocity, total impulse, specific impulse, mass ratio and propellant fraction, energy and efficiencies, variable thrust | 2.1 (26), 2.2 (31), 2.3 (33), 2.4 (35), 2.5 (38), 2.7 (40) | `propulsion::performance` (unit-bearing types) | Rust | — |
| 3 | 4 Flight Performance (99–153) | The rocket equation with gravity and drag losses, forces on a vehicle in the atmosphere, planar ascent equations of motion, orbits and mission velocity, maneuvers and RCS, multistage vehicles and staging, flight stability; Appendix 2 standard atmosphere as its table | 4.1 (99), 4.2 (104), 4.3 (106), 4.7 (136), 4.4 (113), 4.5 (127), 4.6 (133), 4.9 (147), App. 2 (747) | `flight::ascent`, `flight::staging`, `atmosphere::us_standard` | Rust; CUDA for dispersed ascent Monte Carlo | NR 3 (tables), NR 17 (ODE), Wie 4.1 |
| 4 | 6 Liquid Propellant Rocket Engine Fundamentals (189–243) | Feed systems and pressure budgets, tank pressurization mass, turbopump feed systems and **engine cycles** (gas generator, staged combustion, full-flow staged combustion), engine families, valves and lines | 6.6 (217), 6.3 (203), 6.5 (212), 6.4 (205), 6.9 (233), 6.2 (196), 6.7 (229) | `propulsion::cycle` (lumped-parameter thermofluid network) | Rust | NR 9.6–9.7 (nonlinear network solve), rank 1, 5, 6 |
| 5 | 10 Turbopumps and Their Gas Supplies (365–398) | Pump head, NPSH and cavitation, specific speed, turbine work and efficiency, shaft power and pressure balances, gas generators and preburners | 10.4 (376), 10.5 (378), 10.6 (387), 10.8 (393), 10.7 (390), 10.3 (371) | `propulsion::turbopump` | Rust | rank 4 |
| 6 | 5 Chemical Rocket Propellant Performance Analysis (154–188) | Thermochemistry of the chamber and the expansion: adiabatic flame temperature, frozen versus shifting equilibrium, Gibbs minimization, results for LOX/methane and others | 5.1 (156), 5.2 (161), 5.3 (166), 5.4 (171), 5.5 (172) | `propulsion::thermochemistry` with Cantera as the golden-vector emitter | Rust; Cantera (Python) as reference | NR 9.6, `cantera_stuff/`, `Surrogates/` |
| 7 | 11 Engine Systems, Controls, and Integration (399–433) | Propellant budget, engine design closure, start and shutdown transients, automatic and computer control, calibration, health monitoring | 11.4 (412), 11.5 (423), 11.1 (399), 11.3 (403), 11.2 (401), 11.6 (430) | `propulsion::engine_control`, `propulsion::calibration` | Rust | rank 4, NR 15 (fitting) |
| 8 | 8 Thrust Chambers (271–343) | Injector orifice flow and pressure drop, chamber volume and characteristic length, heat transfer (Bartz), regenerative cooling hydraulics and wall temperature, wall loads, starting and ignition, a full sample design | 8.5 (310), 8.2 (285), 8.1 (276), 8.9 (328), 8.6 (322), 8.3 (300) | `propulsion::thrust_chamber`, `propulsion::cooling` | Rust; CUDA for transient wall conduction | NR 20.2 (diffusion), rank 1 |
| 9 | 9 Liquid Propellant Combustion and Its Stability (344–364) | Combustion zones, acoustic modes of a chamber, the Rayleigh criterion, rating techniques and remedies | 9.3 (349), 9.1 (344), 9.2 (348) | `combustion::chamber_acoustics` | Rust | NR 6.5 (Bessel), NR 13 (spectra), `CombustionInstability/`, Lieuwen and Natanzon (parsed) |
| 10 | 18 Thrust Vector Control (671–689) | Gimbal, jet vanes, secondary injection; actuator limits and bandwidth; integration with the vehicle | 18.1 (673), 18.4 (687), 18.2 (683), 18.3 (686) | `actuators::gimbal` | Rust | Wie 9.1 |
| 11 | 17 Electric Propulsion (620–669) | Ideal performance and optimum specific impulse for a mission, electrostatic thruster relations, power supplies; the Starlink Hall-thruster case | 17.1 (626), 17.4 (654), 17.3 (638), 17.6 (661), 17.5 (658) | `propulsion::electric` | Rust | rank 2 |
| 12 | 20 Rocket Exhaust Plumes (703–725) | Plume structure, impingement forces and heating on structures, radio attenuation | 20.2 (717), 20.1 (705), 20.3 (723) | `propulsion::plume_impingement` (small) | Rust | — |
| 13 | 7 Liquid Propellants (244–270) | Property data for LOX, methane and hydrocarbons, hydrogen, hydrazines; hazards and specifications | 7.2 (255), 7.3 (259), 7.1 (245), 7.4 (264) | `propellants::properties` (tables; `Liquiddensity.py`, `LiquidVaporEq.py` exist) | Rust | NR 3 |
| 14 | 12 Solid Propellant Rocket Motor Fundamentals (434–490) | Internal ballistics: burn rate law, mass balance, chamber pressure equilibrium, erosive burning, grain regression | 12.1 (439), 12.3 (462), 12.2 (457) | `propulsion::solid_ballistics` | Rust | NR 17 |
| 15 | 21 Rocket Testing (726–742) | Test types, instrumentation and data management, health monitoring, flight testing, post-accident procedures | 21.3 (735), 21.1 (726), 21.4 (739) | none; informs the telemetry and test-infrastructure work | — | read-only |
| 16 | 13, 14, 15 Solid propellants, combustion, components (491–592) | Solid-motor materials, stability, and hardware | 14.4 (543) | none | — | read-only |
| 17 | 16 Hybrid Propellant Rocket Propulsion (593–619) | Regression-rate ballistics | 16.2 (599) | none | — | read-only |
| 18 | 1, 19 Classification; Selection (1–25, 690–702) | Landscape and selection criteria | 1.2 (4), 19.2 (697) | none | — | read-only |

## 2. What the ranks build, in order

- **Ranks 1–2** give the ideal-rocket performance model with unit-bearing
  types and every Appendix-3 identity as a test.
- **Rank 3** turns it into a vehicle: ascent with losses and staging, with
  the standard atmosphere as an interpolation table. With the NR integrators
  and Wie's attitude half, this is the 6-DOF ascent simulation.
- **Ranks 4–6** are the engine: a lumped-parameter network of tanks, pumps,
  turbines, preburners, chamber, and lines closed by a Newton solve, with
  thermochemistry from Cantera-verified tables. This is the model behind
  Raptor-class simulation and data analysis.
- **Rank 7** adds transients, calibration, and health monitoring, which is
  where post-flight and test data meet the model.
- **Ranks 8–10** are the components that couple into the vehicle: chamber
  heat transfer and cooling, combustion acoustics, and TVC actuators.
- **Ranks 11–14** are secondary models; 15–18 are read-only.

## 3. Coding briefs

### 3.1 `propulsion::nozzle` (chapter 3) — Rust

- **Physics.** Steady, one-dimensional, isentropic flow of a calorically
  perfect gas through a converging–diverging nozzle. Mass flow is fixed by the
  throat (sonic) condition; exit velocity follows from enthalpy conservation
  between chamber and exit; thrust is momentum flux plus the pressure term;
  the thrust coefficient C_F and characteristic velocity c* separate the
  nozzle's contribution from the chamber's. The area-ratio relation is
  monotone on each branch and inverted by root finding. Real nozzles add
  divergence loss, boundary-layer and two-phase losses, and separation when
  overexpanded (the Summerfield-type criterion the book states).
- **Named objects.** `SpecificHeatRatio`, `ChamberConditions { p_c, T_c, R }`,
  `AreaRatio`, `PressureRatio`, `ThrustCoefficient`,
  `CharacteristicVelocity`, `NozzleGeometry::{Conical { half_angle }, Bell {
  fractional_length }}`, `DivergenceLoss`, `SeparationCriterion`,
  `IdealNozzle::performance(ambient_pressure) -> NozzlePerformance`.
- **Tests (from Appendix 3 and the mathematics).** Throat Mach = 1; mass flow
  from c* equals mass flow from the throat state; C_F at optimum expansion
  equals the book's closed form; C_F is maximal at p_e = p_a for fixed ε and
  γ; area ratio inversion round trips on both branches; limits as ε → 1 and
  as p_a → 0; divergence loss for a 15° cone matches the half-angle formula;
  golden vectors emitted from `NozzleTheory.py` (Python is the emitter here,
  the same discipline as C++ twins).

### 3.2 `propulsion::performance` (chapter 2) — Rust

- **Physics.** F = ṁ c with c the effective exhaust velocity; I_sp = c / g₀;
  total impulse I_t = ∫ F dt; the ideal rocket equation Δv = c ln(m₀/m_f);
  propellant mass fraction and mass ratio; internal, propulsive, and overall
  efficiencies.
- **Named objects.** `Thrust`, `MassFlow`, `EffectiveExhaustVelocity`,
  `SpecificImpulse`, `TotalImpulse`, `MassRatio`, `PropellantMassFraction`,
  `RocketEquation`. Units in the type; no bare `f64` at a public boundary.
- **Tests.** Dimensional identities; Δv monotone in mass ratio; the
  efficiencies lie in [0, 1]; the book's typical performance values
  (§2.6) reproduced within stated rounding.

### 3.3 `flight::ascent` + `atmosphere::us_standard` (chapter 4, Appendix 2) — Rust, CUDA later

- **Physics.** Planar ascent of a point mass with time-varying mass:
  thrust along the body axis, drag from the atmosphere table, gravity from
  the two-body field, pitch program as an input; losses accounted as
  ∫ g sin γ dt and ∫ D/m dt; staging as a discontinuity in mass and thrust.
  The standard atmosphere is a table of pressure, temperature, density
  versus altitude interpolated with the NR chapter 3 tools.
- **Named objects.** `StandardAtmosphere` (table + interpolation kind as a
  type), `AscentState { position, velocity, mass }`, `Stage { thrust, isp,
  structural_mass, propellant_mass }`, `PitchProgram`, `AscentEquations` as
  an ODE right-hand side, `LossAccounting`.
- **Tests.** Gravity-free drag-free flight matches the rocket equation
  exactly; vertical ascent in vacuum matches the closed form with constant
  gravity; atmosphere interpolation reproduces the table at nodes; staging
  identities (§4.7); energy accounting closes to integrator tolerance.
- **CUDA.** Dispersed ascent (thrust, Isp, drag coefficient, winds later) is
  the propulsion Monte Carlo kernel; the Rust ODE is the reference.

### 3.4 `propulsion::cycle` + `propulsion::turbopump` (chapters 6 and 10) — Rust

- **Physics.** An engine is a network: tanks at pressure, lines with
  pressure drops proportional to ṁ², pumps that add head at a power cost,
  turbines driven by preburner or gas-generator flow, a chamber whose
  pressure follows from c* and throat area. Steady operation is the
  simultaneous solution of the mass, pressure, and power balances, a
  nonlinear system solved by globally convergent Newton (NR 9.7). Cycle
  variants differ in which flows drive the turbines: gas generator (open),
  staged combustion (closed, one preburner), full-flow staged combustion
  (closed, oxidizer-rich and fuel-rich preburners driving separate pumps).
  Pump similarity laws and NPSH bound the operating point.
- **Named objects.** `Tank`, `Line`, `Pump { head_curve, efficiency }`,
  `Turbine`, `Preburner`, `Chamber`, `EngineCycle::{GasGenerator,
  StagedCombustion, FullFlowStagedCombustion}`, `PowerBalance`,
  `PressureBudget`, `OperatingPoint`, `TurbopumpDesignPoint`.
- **Tests.** Mass, pressure, and power balances close at the solved point to
  solver tolerance; the book's worked engine numbers (§6.6, §10.4) are
  reproduced; NPSH margin is positive at the design point; affinity laws
  hold under speed scaling; a full-flow cycle's two pump powers sum to the
  two turbine powers.

### 3.5 `propulsion::thermochemistry` (chapter 5) — Rust with Cantera as emitter

- **Physics.** Chamber composition and temperature from adiabatic,
  isobaric equilibrium (Gibbs minimization under element conservation);
  expansion either frozen or in shifting equilibrium; c*, C_F, and I_sp from
  the resulting properties. The element-conservation projection is the same
  linear constraint used in `Surrogates/` and `SourceTermSurrogate.tex`.
- **Named objects.** `Species` with NASA polynomial thermodynamics,
  `Mixture`, `ElementMatrix`, `EquilibriumSolver` (small-species Gibbs
  minimization), `ExpansionModel::{Frozen, Shifting}`.
- **Tests.** Element conservation to machine precision; agreement with
  Cantera (`cantera_stuff/LOXmeth_eq.py` extended to emit golden vectors) on
  adiabatic flame temperature, composition, c*, and I_sp for LOX/methane at
  Raptor-class pressures within 0.1%; frozen ≤ shifting I_sp always.

### 3.6 `propulsion::thrust_chamber` + `propulsion::cooling` (chapter 8) — Rust, CUDA for transients

- **Physics.** Gas-side heat flux from the Bartz correlation, wall conduction,
  coolant-side convection in channels, coolant temperature rise and pressure
  drop; injector orifice flow ṁ = C_d A √(2 ρ Δp); chamber volume from
  characteristic length L*.
- **Named objects.** `BartzCorrelation`, `WallMaterial`, `CoolingChannel`,
  `RegenerativeJacket`, `InjectorOrifice`, `ChamberSizing`.
- **Tests.** Heat balance closes (gas-side flux = wall conduction = coolant
  pickup) at steady state; the sample design in §8.9 is reproduced to the
  book's rounding; transient wall conduction (NR 20.2) matches the
  semi-infinite-solid closed form for short times.

### 3.7 `combustion::chamber_acoustics` (chapter 9) — Rust

- **Physics.** Acoustic modes of a cylindrical chamber (longitudinal,
  tangential, radial) from the wave equation with Bessel-function radial
  solutions; the Rayleigh criterion for growth when heat release is in phase
  with pressure.
- **Tests.** Mode frequencies match the closed-form cylinder formulas for
  the book's example; the first tangential mode's Bessel root is the
  tabulated value; ties to the parsed Lieuwen and Natanzon equations by tag.

## 4. Reading order for the first month

1. §2.1–2.3 and Appendix 3, then §3.3 with `NozzleTheory.py` open; write
   `propulsion::performance` and `propulsion::nozzle` with the Appendix-3
   tests. 2. §3.4–3.5 real-nozzle corrections. 3. §4.1–4.3 and Appendix 2;
   the ascent ODE on the DOPRI5 port. 4. §4.7 staging. 5. §6.6 engine cycles,
   then §10.4 power balance; the network model. 6. §5.1–5.3 with
   `cantera_stuff/`; golden vectors. 7. §11.4–11.5 start transients and
   calibration. 8. §8.5 Bartz and cooling. Each step ends with the seven
   artifacts.
