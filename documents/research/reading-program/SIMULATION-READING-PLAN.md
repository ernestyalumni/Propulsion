# Reading plan: toward simulating a liquid rocket engine and a solid rocket booster

**Written:** 2026-09-24. **Books:** six parsed, source-reviewed corpora under
`Data/Public/books/EngineeringPhysics/`: Sutton 9e, Hill & Peterson 2e,
Huzel & Huang (1992), Humble (1995), Turns 3e, Williams 2e. Master context:
`PROPULSION-CORPUS.md` in that directory. It has the topic crosswalk and the
folio rules.
**Relation to what exists:** this plan *extends*
[`HillPeterson-…/PAIRING-SUTTON.md`](HillPeterson-MechanicsThermodynamicsPropulsion-2e/PAIRING-SUTTON.md).
That pairing still holds. Here it becomes one thread among six books. The
module names and ranks in
[`Sutton-…/ROADMAP.md`](Sutton-RocketPropulsionElements-9e/ROADMAP.md) are kept.
**All page numbers are printed folios** unless marked PDF.
**In the reading room:** the roadmap page (`#roadmap`) shows the section-level overlap of
Sutton, Hill & Peterson, Huzel & Huang and Humble, from
[`FOUR-BOOK-OVERLAP.json`](FOUR-BOOK-OVERLAP.json). Edit that file to change the map.
Above it, the **solid-motor track** (current focus, 2026-09-29) lists the Phase 8–9
simulation steps in build order, each marked built, next or later, with the sections of
Sutton, H&P, Humble, Huzel, **Williams and Turns** behind it. Its source is
[`SOLID-MOTOR-TRACK.json`](SOLID-MOTOR-TRACK.json). Williams and Turns are now reading-room
books; their ranked roadmaps are in `Williams-CombustionTheory-2e/` and
`Turns-IntroductionToCombustion-3e/`.

## 0. What we are building, and what each book is for

There are two simulators on one shared core.

- **Liquid engine (LRE).** A lumped thermofluid network: tanks → lines →
  pumps and turbines → injector → chamber → nozzle. It is closed by a steady
  power/pressure balance, then run as a transient (start, shutdown,
  throttle). Thrust-chamber heat transfer and regenerative cooling are
  coupled in. Thermochemistry comes from equilibrium tables, with finite-rate
  kinetics where they matter.
- **Solid booster (SRM).** Grain burn-back geometry → internal ballistics
  (lumped, then quasi-1-D port flow with erosive burning) → nozzle with
  two-phase loss → thrust curve → vehicle ascent.
- **Shared core.** Quasi-1-D gas dynamics, equilibrium thermochemistry, 0-D
  reactors with a stiff integrator, isentropic nozzle and C_F, convective
  heat transfer, ODE integration (Cosmos already has DOPRI5; stiff chemistry
  needs NR 17.5-type methods).

| Book | Its role in this plan |
|---|---|
| **Sutton** | The spine and the handbook. It covers every subsystem at design depth and has the most current data. Read it for *what* and *how much*. |
| **Hill & Peterson** | The derivations. Control-volume thrust, quasi-1-D flow, boundary layers, turbomachinery, and solid-chamber stability are derived rather than stated. Read it for *why*, and for what an implementation may assume. |
| **Huzel & Huang** | The liquid-engine design manual. Its chambers, cooling, turbopumps, controls and **§10.2 engine-system dynamic model** are the closest thing in the library to an LRE simulation spec. Its worked sample calculations are test vectors. |
| **Humble** | The system designer's view. It gives sizing procedures, trade studies and two worked case studies (liquid §5.5 and solid §6.6) that make good end-to-end golden vectors. It also has the clearest solid-motor *performance-prediction* chapter (§6.5). |
| **Turns** | The combustion primer: thermochemistry, kinetics, mechanisms (H₂–O₂, CH₄), reactor models, droplets. Read it *before* Williams. |
| **Williams** | The combustion theory: conservation equations for reacting flow, reacting nozzle flow and rocket I_sp, **solid-propellant deflagration (ch. 7)**, SRM acoustic instability (ch. 9), spray combustion, kinetics review. |

## 1. Where you are now

You have finished Sutton Ch. 1 and are in Ch. 2. The notes already cover
force as a one-form, Sutton §2.1 annotated, and the isentropic nozzle.

## 2. The phases

Each phase lists **read** (in order), **be able to derive**, **build**, and
**write** (a shared topic in `documents/notes/topics/`). A → marks the order
within a sitting. Phases 1–3 are shared. Phases 4–7 are the liquid engine.
Phases 8–9 are the solid booster. Phase 10 puts either one on a vehicle.
After Phase 3 you can run the liquid track (4–7) and the solid track (8–9)
in either order. The solid track is shorter.

### Phase 1 — Thrust and the ideal rocket (shared)

- **Read:** Sutton §2.2–2.7 (31–44) → H&P §1.2 (4) and §1.3 (8) → Humble §3.3
  thrust equations, C_F and c* (107–120) → Huzel §1.2–1.4 (4–17), the
  engineer's summary including correction factors (16). Then H&P **Ch. 3
  entire** (65–92) → Sutton Ch. 3 (45–98) → H&P §11.3 nozzles (520) →
  Humble §3.2.5 (95) and §3.4 heat addition (120–127).
- **Derive:** F = ṁv_e + (p_e − p_a)A_e as a control-volume consequence (the
  step the force-one-form note stops short of); c* and C_F as the
  chamber/nozzle split; the area–Mach relation on both branches; Rayleigh
  and Fanno lines from H&P's single set of equations.
- **Build:** `propulsion::performance`, `propulsion::nozzle` (Sutton ranks
  1–2). Port `NozzleTheory.py`.
- **Write:** extend `nozzle.tex` with C_F, under- and over-expansion and
  the separation criterion. Add a thrust-equation topic that continues
  `force-one-form.tex`.

### Phase 2 — Equilibrium thermochemistry (shared)

- **Read:** Turns Ch. 2 (12–78): adiabatic flame temperature, equilibrium →
  H&P §2.4 (40) → Humble Ch. 4 (149–178), especially §4.3.1 equilibrium
  constant (162) and §4.3.2 free-energy minimization (166) → Sutton Ch. 5
  (154–188), frozen versus shifting expansion (§5.3, 166) → Williams App. A
  (521ff) for the statistical-mechanics statement.
- **Derive:** equilibrium as Gibbs minimization under element constraints.
  The Lagrange multipliers are the element potentials. The two methods
  (K_p versus minimization) give the same stationarity conditions.
- **Build:** `propulsion::thermochemistry`. Gibbs minimizer with Newton
  iteration; Cantera (`cantera_stuff/`, `Surrogates/.venv`) and CEA-style
  outputs as golden vectors. Sutton rank 6.
- **Write:** topic `equilibrium.tex`.

### Phase 3 — Kinetics and reactors: the "every reaction rate" level (shared)

- **Read:** Turns Ch. 4 (107–148): elementary rates, steady-state and
  partial-equilibrium approximations, chemical time scales (129) → Turns
  Ch. 5 (149–182): H₂–O₂ (149), CH₄ (159), reduced mechanisms → Turns
  Ch. 6 (183–219): constant-p/V reactors, well-stirred reactor (194),
  plug-flow reactor (206) → Williams App. B (554–603): mass action, chain
  branching, thermal explosion, Arrhenius and transition-state theory →
  Williams §4.1–4.2 (92–107): ignition delay, **reacting quasi-1-D nozzle
  flow and rocket I_sp (99)**, near-equilibrium versus near-frozen (100) →
  H&P §12.4 nonequilibrium expansion (578) → Humble §4.5 (172).
- **Derive:** the reactor ODEs from Turns Ch. 6. The stiffness ratio from
  the Jacobian's eigenvalues. Frozen and shifting expansion as the two
  Damköhler limits of Williams §4.2.3.
- **Build:** 0-D reactor + stiff integrator (NR 17.5); finite-rate nozzle
  march. This connects directly to the Surrogates E1 stiff-chemistry
  benchmark (measured H₂ stiffness 10⁵–10⁸).
- **Write:** topic `frozen-shifting.tex`: the Damköhler-number criterion
  for when Sutton's two bracketing answers apply.

### Phase 4 — LRE I: the thrust chamber and its cooling

Huzel Ch. 4 is the spine. Take the boundary layer from H&P *before* any
correlation.

- **Read:** Huzel §4.1–4.3 (67–84): performance, chamber volume and L*,
  nozzle shape → Sutton §8.1–8.2 (276–300) → H&P **Ch. 4 entire** (93–137)
  → Sutton §8.5 heat transfer (310) → Huzel §4.4 cooling (84–104): gas side
  (85), regenerative (88), coolant side (89), passage pressure drop (93),
  film, ablative, radiation → H&P §11.4 (541) → Humble §3.5 (128–138) and
  §5.4.1 thrust chamber (217–244) → Huzel §4.5 injector orifice sizing
  (104–115) → Sutton §8.9 sample design (328).
- **Derive:** Bartz as a turbulent-boundary-layer/Reynolds-analogy result
  (the keystone pairing). The wall temperature balance across the gas film,
  the wall and the coolant.
- **Build:** `propulsion::thrust_chamber`, `propulsion::cooling`: a 1-D
  marching regen-channel model. **Test vectors:** Huzel's sample
  calculations in Ch. 4 and Sutton §8.9.
- **Write:** topic `bartz.tex`.

### Phase 5 — LRE II: feed systems, turbomachinery, the steady cycle balance

- **Read:** Sutton Ch. 6 (189–243), especially §6.6 cycles (217) → Huzel
  Ch. 3 (53–66), the four sample engines A-1…A-4 with their cycles and
  start/cutoff sequences → Huzel Ch. 5 pressurization (135–154) → Huzel
  Ch. 6 turbopumps (155–218): inducers (175), centrifugal pumps (179),
  turbines (194) → H&P Ch. 13 (615–649), with Ch. 7–9 behind it only if a
  real turbomachinery model is wanted → Sutton Ch. 10 (365–398) → Humble
  §5.3–5.4 (194–282).
- **Derive:** pump head and NPSH, the turbine work equation, and the power
  balance that closes a gas-generator and a staged-combustion cycle.
- **Build:** `propulsion::cycle` (Sutton rank 4) and `propulsion::turbopump`
  (rank 5): nodes and branches, closed by Newton (NR 9.6–9.7). **Test
  vectors:** Huzel's A-1…A-4 engines; Humble §5.5 case study (282–294).
- **Write:** topic `cycle-balance.tex`.

### Phase 6 — LRE III: dynamics, control, calibration (the engine simulation)

- **Read:** Huzel **§10.2 Engine System Dynamic Analysis (346–350)**: the
  model equations, start/shutdown transients, engine–vehicle interaction,
  low-frequency instability → Huzel §10.3–10.4 (350–356): calibration,
  influence coefficients, nonlinear corrections → Huzel Ch. 7 (219–284):
  control laws (228), valves and regulators, instrumentation, condition
  monitoring, **post-flight data analysis (279)** → Sutton Ch. 11 (399–433).
- **Derive:** the lumped volume (capacitance) and line (inertance)
  equations. Influence coefficients as the Jacobian of the steady balance.
- **Build:** a transient engine model from the Phase 5 network plus
  capacitances and inertances; a start-sequence simulation; calibration by
  fitting (NR 15). Sutton rank 7.
- **Write:** topic `engine-transients.tex`.

### Phase 7 — LRE IV: combustion in the chamber, and stability

- **Read:** Turns Ch. 10 droplets (366ff), including the liquid-rocket
  application (371) → Williams Ch. 3 diffusion flames and droplet burning
  (38–91) → Williams Ch. 11 spray combustion (446–484) → Sutton Ch. 9
  (344–364) → Huzel §4.8 instability (127–134) → H&P §12.5 (581) and §12.8
  (606) → Williams Ch. 9 (294–372) → Lieuwen and Natanzon (parsed) for depth.
- **Build:** chamber acoustic modes (Sutton rank 9). A chug model that
  couples the Phase 6 feed dynamics to a combustion time lag.

### Phase 8 — SRM I: propellant, burning rate, internal ballistics

- **Read:** Sutton **Ch. 12** (434–490): burning rate (439), mass balance,
  temperature sensitivity, erosive burning, grain configurations (462),
  grain stress (472) → H&P §12.6–12.7 (589–605): two-phase loss (598),
  chamber-pressure ODE, stability, port pressure drop → Humble **Ch. 6**
  (295–364): §6.4 propellants and burning rate (323–331), **§6.5
  performance prediction (331–351): lumped-parameter (334) and spatial
  pressure variation (342)** → Sutton Ch. 13 (491–535) for ingredients.
- **Derive:** already written. See the new topic
  [`solid-ballistics.tex`](../../notes/topics/solid-ballistics.tex): the
  lumped ODE, equilibrium p = (K a ρ_p c*)^{1/(1−n)}, the n < 1 stability
  proof with time constant L*/((1−n)Γ²c*), temperature sensitivity, and
  port pressure drop to its choking limit.
- **Build:** `propulsion::solid_ballistics` (Sutton rank 14, promoted).
  (a) grain burn-back geometry, A_b(y) and V(y): analytic for a tube or
  rod-and-tube, level-set or fast-marching for star and finocyl (the
  data-parallel, CUDA-worthy piece); (b) lumped (p, y) ODE; (c) quasi-1-D
  port march with erosive burning (Sutton Eq. 12-17). **Test vector:**
  Humble §6.6 case study (352–360).
- **Write:** extend `solid-ballistics.tex` with the quasi-1-D port equations.
- **Status (2026-09-26):** first slice built in C++ and Rust (twins, golden
  vectors bitwise identical): lumped (y, m_c) ballistics, tubular and
  end-burning grains, both temperature laws (Sutton 12-12, Williams 7-41),
  lumped aft-end erosive burning, burnout landed exactly, and the
  **gas-generator mode**: grain → orifice → plenum → N pintle valves
  (Sutton Fig. 12-27; Huzel pp. 116, 148–149; Turns §6 plenum). See
  [`derivations/SolidRocketMotorGasGenerator.md`](../../derivations/SolidRocketMotorGasGenerator.md).
  Still open: star/finocyl level-set burn-back, quasi-1-D port march, the
  Humble §6.6 test vector, closed-loop valve control.

### Phase 9 — SRM II: combustion physics, hardware, and the limits of fidelity

- **Read:** Williams **Ch. 7** (229–264): homogeneous deflagration (230),
  condensed- versus gas-phase control (238, 243), heterogeneous propellants
  (251), erosive burning (258) → Sutton Ch. 14 (536–554): ignition,
  extinction, instability → Williams §9.1 (295ff): SRM acoustic modes,
  admittance, damping including particle damping (312), combustion response
  (319) → Turns Ch. 14 (527–555): carbon one-film and two-film models, the
  analogue for a burning metal particle → Sutton §3.5 two-phase nozzle flow
  → Sutton Ch. 15 (555–592): case, nozzle and throat erosion, igniter →
  Humble §6.3 sizing (306–323).
- **Build:** throat-erosion and two-phase-loss corrections; an ignition
  transient using the full pressure ODE; optionally a Williams-§7.5
  gas-phase-controlled burning-rate model, to see how far theory gets from
  a measured (a, n).

### Phase 10 — On a vehicle

- **Read:** Sutton Ch. 4 (99–153) → H&P §10.3–10.6 and App. VIII staging
  (729) → Humble §2.6 Earth to orbit, steering, flight-simulation programs
  (61–76) and App. C staging (715) → Sutton Ch. 18 TVC (671–689) with Humble
  §6.3.8 (320) and Huzel §9.7 gimbal mounts (340).
- **Build:** `flight::ascent`, `flight::staging` (Sutton rank 3) driven by
  the Phase 6 engine or the Phase 8 thrust curve.

### Deferred (read only when a model needs it)

Huzel Ch. 8–9 (tanks, ducts, bellows): structural; §9.1 pump-inlet line
vibration only for pogo. Humble Ch. 7–9, 11. H&P Ch. 5–6 (air-breathing).
Turns Ch. 7–9, 11–13, 15–17. Williams Ch. 5–6, 8, 10, 12. Williams Ch. 8
(ignition and extinction) is worth taking with Phase 9 if the ignition
transient becomes a focus.

## 3. How much chemistry does a solid motor carry? (the short answer)

The full argument, with citations, is the last subsection of
`documents/notes/topics/solid-ballistics.tex`.

- In a solid-motor design or performance model, chemistry enters in
  **two places only**: (1) equilibrium thermochemistry gives T₀, γ, c* and
  C_F, including condensed Al₂O₃ in the products (the same CEA-type
  calculation as a liquid engine); (2) the burning-rate law r = a pⁿ, whose
  a and n are **measured** (strand burner → ballistic test motor →
  full-scale firing) and not computed from kinetics. Sutton says it
  directly (p. 445–446): analytical modeling has "yet to adequately predict
  the burning rate of a new propellant in a new rocket motor."
- **Why:** a composite propellant burns in a thin (µm to sub-mm),
  heterogeneous zone. Oxidizer crystals sit in a binder; there is
  condensed-phase decomposition, a surface, and premixed and diffusion
  flames whose layout follows the particle packing. The gas-phase flames
  can carry detailed elementary mechanisms, and research codes do this for
  AP and nitramines. The condensed phase is only ever a few global steps.
  The packing is statistical. The flame is about 10⁴–10⁶ times thinner than
  the motor. Full-motor CFD therefore imposes r(p, local flow, T_b) as a
  boundary condition. The engineering flame models (Beckstead–Derr–Price
  multiple-flame model) are semi-empirical.
- **Aluminium** adds a second empirical layer: agglomeration on the
  surface, droplet burning in the port, and alumina that causes two-phase
  nozzle loss and acoustic damping.
- **The working fidelity** of solid-motor design is quasi-1-D internal
  ballistics with measured burning rates, erosive-burning correlations and
  burn-back geometry, plus two-phase and finite-rate nozzle corrections.
  Multidimensional and coupled fluid–structure–combustion simulation exists
  for research and failure analysis. Resolved-flame chemistry is confined
  to propellant-scale research, not whole motors.
- **Correction to the liquid side:** a full liquid engine is not simulated
  with every elementary rate everywhere either. Detailed kinetics are used
  in 0-D and 1-D problems (reactors, flamelets, finite-rate nozzles, Phase 3)
  and are tabulated or reduced for chamber CFD. The real asymmetry is that
  a liquid engine's chemistry *can* be computed from known gas-phase
  mechanisms where it matters, while a solid's burning surface still has
  to be measured.

## 4. Milestones

| # | Milestone | Phases | Golden vectors |
|---|---|---|---|
| M1 | Ideal-rocket calculator (c*, C_F, I_sp, area ratio) | 1 | Sutton App. 3 identities, Sutton ch. 3 examples |
| M2 | Equilibrium chamber and frozen/shifting nozzle | 2 | Cantera / CEA; Sutton ch. 5 tables |
| M3 | 0-D reactors + stiff integrator + finite-rate nozzle | 3 | Cantera reactors; Surrogates E1 |
| M4 | Regen-cooled chamber wall-temperature march | 4 | Huzel Ch. 4 sample calcs; Sutton §8.9 |
| M5 | Steady cycle balance (GG, staged combustion) | 5 | Huzel A-1…A-4; Humble §5.5 |
| M6 | **Transient liquid engine** | 6 | Huzel §10.2 model structure |
| M7 | **Lumped SRM with grain burn-back → thrust curve** | 8 | Humble §6.6 |
| M8 | Quasi-1-D SRM with erosive burning | 8–9 | H&P Eq. (12.37) limit; Sutton Eq. 12-17 |
| M9 | Ascent with SRM booster(s) and liquid core | 10 | Sutton ch. 4 examples; H&P App. VIII |

Record sections read in the book ledgers as before. The Huzel, Humble,
Turns and Williams ledgers and `progress.json` files do not exist yet.
Create them when the first section of each is read, following the four
existing books.
