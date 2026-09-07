# Hill and Peterson, *Mechanics and Thermodynamics of Propulsion*, 2nd ed. — reading roadmap and coding briefs

**Status:** written 2026-09-06 from the verified table of contents, while the
OCR of the full text runs. PDF page = printed page + 12 throughout the body;
the front matter carries roman folios equal to the PDF page. `INDEX.md` in the
corpus slug has every section with both numbers.
**Corpus slug:** `Data/Public/books/EngineeringPhysics/HillPeterson-MechanicsThermodynamicsPropulsion-2e/`
**Edition note:** Addison-Wesley 1992, ISBN 0-201-14659-2, "reprinted with
corrections June 1992" (title and copyright pages verified from the scan). The
source PDF is a 300 dpi bilevel **scan with no text layer**, unlike every other
book in this program; see `../README.md` for what that changed about parsing.
**Why this book:** it is the derivation behind Sutton. Sutton states the
result and gives the design data; Hill and Peterson start from a control
volume and get there, so you can see which assumptions were spent. The
book-by-book argument, the pairing table, and a joint reading order are in
[`PAIRING-SUTTON.md`](PAIRING-SUTTON.md) — **read that first**; this file
ranks Hill and Peterson on its own terms and specifies the modules.
**Rules that apply:** stories 15 and 16; the seven-step protocol in
`../../numerical-recipes-rewrite/ROADMAP.md` §2.
**Already in the repository (port, do not rewrite blind):** `NozzleTheory.py`,
`Propulsion.py`, `thermo.py`, `cantera_stuff/`, `ccdroplet/`,
`CombustionInstability/`, `Surrogates/`.

## 1. Chapter ranking

Tier 0: §1.2–1.3 (4–13) for the vocabulary, and Appendix X, List of Symbols
(739), which is worth keeping open — this book's notation differs from
Sutton's and from Wie's.

| Rank | Chapter | The physics that earns its place | Sections (printed page) | Becomes | Language | Depends on |
|---|---|---|---|---|---|---|
| 1 | **3** Steady One-Dimensional Flow of a Perfect Gas (65–92) | The whole quasi-1-D family derived from one set of equations: isentropic flow, constant-area flow with stagnation-temperature change (Rayleigh), constant-area flow with friction (Fanno), and normal shocks — each with the assumption that produced it | 3.2 (66), 3.3 (69), 3.4 (72), 3.5 (74), 3.6 (77), 3.7 (83) | `gasdynamics::quasi1d` | Rust; C++ twin via golden vectors | NR 9 (root find for the area-ratio and shock inversions) |
| 2 | **11** Chemical Rocket Thrust Chambers (513–567) | Nozzle performance and design beyond the ideal: thrust coefficient and `c*`, two-dimensional and divergence losses, contour design, then rocket heat transfer and regenerative cooling | 11.3 (520), 11.2 (515), 11.4 (541), 11.5 (see `INDEX.md`), 11.1 (513) | `propulsion::nozzle`, `propulsion::rocket_heat_transfer` | Rust (port `NozzleTheory.py`) | rank 1, rank 4 |
| 3 | **2** Mechanics and Thermodynamics of Fluid Flow (23–62) | The control-volume conservation laws the entire book is built on, the perfect-gas and variable-`c_p` thermodynamics, and equilibrium combustion from Gibbs free energy and the equilibrium constant | 2.2 (24), 2.4 (40), 2.3 (32) | `thermo::control_volume`, `thermo::equilibrium` | Rust; Cantera as golden-vector emitter | NR 9.6 (Gibbs minimization) |
| 4 | **4** Boundary Layer Mechanics and Heat Transfer (93–137) | The boundary-layer equations, exact solutions (Blasius, Howarth), the momentum-integral method with Thwaites' relation and its separation criterion, the turbulent boundary layer, and Reynolds analogy between skin friction and heat transfer | 4.2 (101), 4.3 (107), 4.4 (111), 4.5 (124), 4.1 (93) | `fluids::boundary_layer` | Rust; CUDA only if a swept parameter study wants it | NR 17 (ODE for Blasius), NR 4 (quadrature for the Thwaites integral) |
| 5 | **10** Performance of Rocket Vehicles (469–512) | Vehicle acceleration with gravity and drag, the chemical-rocket performance limits, optimum specific impulse for a power-limited electrical vehicle, and mission Δv; Appendix VIII derives the multistage optimum | 10.3 (472), 10.6 (495), 10.4 (478), 10.5 (490), 10.2 (470); App. VIII (729) | `flight::rocket_performance`, `flight::staging` | Rust | rank 1, NR 17, NR 10 (constrained optimum) |
| 6 | **13** Turbomachinery for Liquid-Propellant Rockets (615–649) | The rocket turbopump proper: centrifugal pump stage, inducers and the cavitation/NPSH limit, and the drive turbine — the chapter Sutton §10 compresses into a handbook summary | 13.2 (621), 13.3 (630), 13.4 (640), 13.1 (615) | `propulsion::turbopump` | Rust | ranks 8, 9, 10 |
| 7 | **12** Chemical Rocket Propellants: Combustion and Expansion (569–613) | Equilibrium composition of the products, non-equilibrium (frozen versus shifting) expansion as a *rate* problem, liquid-propellant chamber atomization and vaporization time scales, solid-propellant burning-rate law and chamber-pressure stability | 12.3 (572), 12.4 (578), 12.5 (581), 12.6 (589), 12.7 (598), 12.8 (606) | `propulsion::thermochemistry`, `propulsion::solid_ballistics` | Rust; Cantera as emitter | rank 3, `cantera_stuff/`, `Surrogates/` |
| 8 | **7** Axial Compressors (275–364) | Euler's work equation and angular momentum, velocity triangles, stage and multistage characteristics, diffusion limits and stall, compressor efficiency, degree of reaction, radial equilibrium, and an explicit preliminary-design procedure | 7.2 (277), 7.3 (282), 7.4 (288), 7.6 (303), 7.8 (330), 7.9 (332), 7.10 (336) | `turbomachinery::axial_compressor` | Rust | rank 1, rank 4; App. IV (705), App. V (713) |
| 9 | **8** Axial Turbines (367–423) | The axial turbine stage and its efficiency, rotor blade and disc stresses, blade cooling, turbine performance and compressor matching, stage design | 8.2 (370), 8.3 (377), 8.7 (402), 8.4 (384), 8.5 (393), 8.8 (406) | `turbomachinery::axial_turbine` | Rust | rank 8; App. VI (717) |
| 10 | **9** The Centrifugal Compressor (425–465) | Centrifugal stage dynamics, the inducer and impeller, the diffuser, and stage design — the direct ancestor of the rocket centrifugal pump in Chapter 13 | 9.2 (427), 9.3 (435), 9.4 (445), 9.6 (453) | `turbomachinery::centrifugal` | Rust | rank 8; App. VII (723) |
| 11 | **14** Electrical Rocket Propulsion (651–684) | Electrostatic acceleration, bombardment ionization, the plane diode and the space-charge (Child–Langmuir) limit, thruster performance, the arcjet, pulsed magnetoplasma accelerators | 14.4 (666), 14.2 (654), 14.5 (671), 14.3 (660), 14.6 (674), 14.7 (679); App. IX (733) | `propulsion::electric` | Rust | rank 5 (§10.5 sets the mission-level optimum) |
| 12 | **5** Thermodynamics of Aircraft Jet Engines (141–216) | Ramjet, turbojet, turbofan, turboprop cycle analysis; thrust and propulsive/thermal efficiency; engine–aircraft matching. No counterpart in Sutton | 5.2 (146), 5.4 (164), 5.5 (177), 5.3 (155), 5.6 (189), 5.8 (202) | `propulsion::airbreathing_cycle` | Rust | rank 1, rank 3 |
| 13 | **6** Aerothermodynamics of Inlets, Combustors, and Nozzles (217–273) | Subsonic and supersonic inlet design (shock systems, buzz), gas-turbine combustors, afterburners, supersonic combustion, exhaust nozzles | 6.3 (226), 6.2 (218), 6.4 (242), 6.7 (264) | none initially; §6.3 exercises `gasdynamics::quasi1d` hard and is its best test case | — | rank 1 |
| 14 | **1** The Jet Propulsion Principle (3–22) | Thrust from the momentum theorem on a control volume; rockets, propellers, turbojets and ramjets as one family | 1.2 (4), 1.3 (8) | none; vocabulary | — | read-only |

### The appendices are code, not prose

Unusually for this program, five of this book's appendices are explicit
step-by-step design procedures with worked cases. They are the cheapest
modules in the whole reading program, because the book supplies both the
algorithm and its own answer as a test vector:

| Appendix | Printed page | What it is | Becomes |
|---|---|---|---|
| I | 689 | Conversion factors and physical constants | test data for the unit-bearing types |
| II | 693 | Gases at low pressures (property tables) | `thermo::gas_tables` (interpolation; NR 3) |
| III | 699 | ICAO Standard Atmosphere | `atmosphere::icao` — pairs with Sutton App. 2 (US Standard) as a cross-check |
| IV | 705 | Preliminary design of a subsonic axial compressor stage | worked procedure → `turbomachinery::axial_compressor` design driver + its golden vectors |
| V | 713 | Preliminary design of a transonic axial compressor stage | same, transonic branch |
| VI | 717 | Preliminary design of an axial turbine stage | `turbomachinery::axial_turbine` design driver |
| VII | 723 | Preliminary design of a centrifugal compressor stage with prewhirl | `turbomachinery::centrifugal` design driver |
| VIII | 729 | Optimization of multistage rockets | `flight::staging` — the Lagrange-multiplier derivation behind Sutton's staging result |
| IX | 733 | Electrostatic and electromagnetic forces | `propulsion::electric` foundations |
| X | 739 | List of symbols | keep open; this book's notation is not Sutton's |

Also: **Answers to Selected Problems (743)**. Combined with the problem
statements at the end of each chapter, these are ready-made acceptance tests
for every module below — the seven-step protocol's "reproduce the book's own
numbers" step comes free here.

## 2. What the ranks build, in order

- **Rank 1** is the foundation and the first module: quasi-1-D compressible
  flow, with isentropic, Rayleigh, Fanno and normal-shock relations as one
  coherent family. Everything in both books that involves a nozzle, an inlet,
  or a blade row stands on it.
- **Ranks 2–3** complete the ideal rocket: the thrust chamber and the
  thermodynamics (control volume, real `c_p`, chemical equilibrium) it runs on.
- **Rank 4** is the keystone shared with Sutton: boundary layers, and with
  them wall heat transfer. It is what makes the Bartz correlation in Sutton
  §8.5 something you understand rather than something you copy.
- **Rank 5** lifts the engine to a vehicle and a mission, with Appendix VIII
  supplying the staging optimum.
- **Ranks 6, 8–10** are the turbomachinery arc, and the largest single gain
  over Sutton: axial compressor → axial turbine → centrifugal compressor →
  rocket turbopump, with Appendices IV–VII as executable design procedures.
- **Rank 7** is propellants and combustion, feeding `Surrogates/` and the
  existing Cantera work.
- **Rank 11** is electric propulsion, derived rather than tabulated.
- **Ranks 12–13** are the air-breathing part, unique to this book; rank 14 is
  read-only vocabulary.

## 3. Coding briefs

Each brief follows the seven-step protocol
(`../../numerical-recipes-rewrite/ROADMAP.md` §2). Written in full for the
first two ranks; the remainder are specified when their rank comes up, so that
the brief is written with the chapter actually read.

### 3.1 `gasdynamics::quasi1d` (chapter 3) — Rust

**The physics.** One-dimensional steady flow of a perfect gas through a duct,
with area change, heat addition, friction, and shocks treated as four
specializations of the same conservation statements (§3.2, printed 66).

**The surface.**

- Isentropic (§3.3): `T0/T`, `p0/p`, `rho0/rho`, `A/A*` as functions of Mach
  number, and their inverses. The `A/A*` inversion is double-valued — subsonic
  and supersonic branches — and the API must make the branch an explicit
  argument, never a guess.
- Rayleigh (§3.5): constant-area flow with stagnation-temperature change,
  including the thermal-choking limit.
- Fanno (§3.6): constant-area flow with friction, `4 f L*/D` as a function of
  Mach number, and the friction-choking limit.
- Normal shock (§3.7): the jump relations and the stagnation-pressure loss.
- Oblique shocks as needed by rank 13 (§6.3).

**Types.** Mach number, stagnation and static states as distinct types; `gamma`
and `R` carried in a `PerfectGas` parameter rather than as free constants. A
function must not be able to take a static temperature where a stagnation one
belongs.

**Tests.** Three independent sources, all of which this book supplies:
(1) the analytic identities (`M = 1` at the throat gives `A/A* = 1`; the shock
relations reduce to identity at `M = 1`); (2) the book's own worked examples
in §3.3–3.7; (3) the end-of-chapter problems (printed 88) against **Answers to
Selected Problems** (printed 743). Round-trip property: `A/A*` forward then
inverted on the named branch returns the original Mach number.

**Cross-check.** Sutton Ch. 3 uses the same relations with rocket notation;
agreement between the two books' worked nozzle examples is the acceptance test
that the module is right and that the notation mapping is right.

**Language.** Rust primary. C++ twin only if the existing stack needs it, via
golden vectors per story 16. No CUDA — this is scalar work; if a swept study
wants throughput, vectorize over the parameter, not over the relation.

### 3.2 `propulsion::nozzle` and `propulsion::rocket_heat_transfer` (chapter 11) — Rust

**The physics.** §11.2 performance characteristics (thrust coefficient,
characteristic velocity, specific impulse and their interrelations); §11.3
nozzles, including the departures from one-dimensionality that set real
performance — divergence loss, two-dimensional effects, contour design;
§11.4 rocket heat transfer and regenerative cooling.

**Depends on** `gasdynamics::quasi1d` for every isentropic relation, and on
`fluids::boundary_layer` (rank 4) for the wall heat flux. Do not implement
§11.4 before Chapter 4 is read — that ordering is the entire argument of
`PAIRING-SUTTON.md`.

**Port, do not rewrite blind.** `NozzleTheory.py` in the repository root
already computes much of §11.2–11.3; it becomes the golden-vector emitter for
the Rust module, and any disagreement is resolved against the book, not
against the Python.

**Tests.** The book's worked examples in §11.2–11.3; the end-of-chapter
problems (printed 559) against the answers (printed 743); and cross-book
agreement with Sutton Ch. 3 on thrust coefficient and area ratio for a stated
chamber condition.

### 3.3 and beyond

Specified when the rank is reached. The order is rank order: `thermo::*` (3),
`fluids::boundary_layer` (4), `flight::*` (5), then the turbomachinery arc
(8 → 9 → 10 → 6), then `propulsion::thermochemistry` (7),
`propulsion::electric` (11), `propulsion::airbreathing_cycle` (12).
