# Wie, *Space Vehicle Dynamics and Control*, 2nd ed. — reading roadmap and coding briefs

**Status:** written 2026-09-02 from the verified table of contents; OCR of the
full text is running (see `README.md` one level up). Section numbers and
printed pages come from the book's own contents; PDF page = printed page + 18
throughout (`INDEX.md` in the corpus slug has every section).
**Corpus slug:** `Data/Public/books/EngineeringPhysics/Wie-SpaceVehicleDynamicsControl-2e/`
**Why this book:** it is the six-degree-of-freedom book. Attitude kinematics,
rigid-body dynamics, attitude control, orbital dynamics and rendezvous, launch
vehicle TVC with slosh, digital control, and structural coupling are all
here, at implementation level, with worked spacecraft examples. It is the
direct counterpart of the "develop 6-DOF dynamics simulation models for GNC
analysis" line in the Starship GNC posting recorded in
`../../numerical-recipes-rewrite/spacex-signal-2026-09-01.md`.
**Rules that apply:** stories 15 and 16; the seven-step protocol in
`../../numerical-recipes-rewrite/ROADMAP.md` §2. Derive from the physics and
the equations; every noun a type; every constant a named parameter; tests
that follow from the mathematics; Rust first.

## 1. One convention warning before anything else

Wie writes quaternions with the scalar component **last** (q₄) and builds the
kinematic differential equation and the direction-cosine matrix on that
layout. The library holds exactly one convention (story 08: Hamilton product,
scalar first, active, body-to-world, `q2*q1` composition). Every formula
transcribed from this book passes through a named adapter that converts
Wie's layout and sign conventions into the library's, and the adapter is
property-tested against the double cover. Do not "fix" the library to match
the book.

## 2. Chapter ranking

Tier 0, read first: §1.3 Dynamic Systems Analysis (71) for the linearization
and stability vocabulary, and §5.4 Quaternions (334) read against
`Cosmos/QuaternionConventionLab/README.md` to fix the adapter.

| Rank | Chapter | The physics that earns its place | Sections (printed page) | Becomes | Language | Depends on |
|---|---|---|---|---|---|---|
| 1 | 5 Rotational Kinematics (323–348) | The attitude state itself: DCM, Euler angles, eigenaxis, quaternion, and the kinematic differential equations that drive the 6-DOF rotational state | 5.1 (323), 5.2 (326), 5.3 (329), 5.4 (334), 5.5 (339) | `attitude::kinematics` with the Wie adapter | Rust; C++ and Rust kernels already exist in the quaternion lab | story 08 |
| 2 | 6 Rigid-Body Dynamics (349–402) | Euler's rotational equations, inertia as a type with principal axes, torque-free motion with closed-form solutions (the test oracle for every attitude propagator), stability about principal axes, gravity gradient in circular orbit, gyrostats and dual-spinners | 6.1 (349), 6.2 (350), 6.3 (355), 6.4 (359), 6.5 (363), 6.6 (366), 6.7 (370), 6.10 (386), 6.11 (396), 6.12 (398) | `attitude::rigid_body`, `attitude::inertia`, `attitude::torque_free_oracle` | Rust; CUDA later for batched Monte Carlo attitude propagation | NR 11 (symmetric eigen), NR 6.11 (Jacobi elliptic), NR 17 (integrators) |
| 3 | 7 Rotational Maneuvers and Attitude Control (403–488) | Quaternion-feedback reorientation with its Lyapunov argument, momentum management with reaction wheels, CMG steering, thruster selection as an allocation problem, pulse-width pulse-frequency modulation | 7.3 (425), 7.4 (444), 7.6 (472), 7.7 (478), 7.5 (461), 7.1 (403), 7.2 (415) | `attitude::control::{quaternion_feedback, momentum_management, jet_selection, pulse_modulation}` | Rust | NR 2.6 (SVD), NR 10.10 (LP), rank 1–2 |
| 4 | 3 Orbital Dynamics (221–276) | Two-body motion, conics, Kepler's time equation, elements ↔ state, perturbations, circular restricted three-body problem and libration points | 3.1 (221), 3.2 (229), 3.3 (233), 3.4 (235), 3.5 (239), 3.6 (246), 3.7 (255), 3.8 (271) | `astrodynamics::{kepler, elements, perturbations, cr3bp}` | Rust; C++ two-body, J2, drag, Numerov exist in `Cosmos/Source/Astrodynamics` | NR 9 (Newton for Kepler), NR 17 |
| 5 | 4 Orbital Maneuvers and Control (277–322) | Launch trajectories and injection, impulsive maneuvers, Hohmann, interplanetary, **rendezvous (Clohessy–Wiltshire)** for docking and refueling, halo orbit determination and control | 4.6 (299), 4.4 (288), 4.3 (286), 4.1 (277), 4.2 (284), 4.5 (290), 4.7 (303) | `astrodynamics::{maneuvers, rendezvous, halo}` | Rust | rank 4, NR 18 (shooting) |
| 6 | 1 Dynamic Systems Modeling and Analysis (3–120) | Lagrange and Hamilton formulations that produce every later equation of motion; linearization, eigenvalue and Lyapunov stability, Routh criteria | 1.3 (71), 1.2 (21), 1.1 (3, skim: exists in wildrider and Cosmos algebra) | `dynamics::{linearize, stability}` | Rust | NR 5.7 (numerical Jacobian), NR 11 |
| 7 | 2 Dynamic Systems Control (121–220) | Digital control (zero-order hold, matrix exponential, discrete equivalents) for a flight computer; state-space methods (controllability, LQR via Riccati); robustness via singular values | 2.4 (152), 2.5 (161), 2.6 (185), 2.2 (123), 2.3 (136), 2.7 (205) | `control::{discretize, lqr, robustness, classical}` | Rust | NR 2.6, NR 11, NR 2 (Schur/Riccati later) |
| 8 | 9 Attitude and Structural Control (531–612) | **TVC design for a launch vehicle with propellant sloshing**, bias-momentum control, flexible-spacecraft stationkeeping, nonlinear pulse-modulated analysis, the Hubble redesign, active vibration control | 9.1 (531), 9.4 (562), 9.2 (542), 9.3 (554), 9.6 (585), 9.5 (574) | `vehicle::{slosh_pendulum, tvc_loop}`, `control::describing_function` | Rust | rank 3, 7, 9 |
| 9 | 8 Structural Dynamics (489–530) | Bars and beams as modal systems, rigid body with beamlike appendages, the flexible-body modes that couple into attitude control and POGO | 8.3 (499), 8.4 (508), 8.2 (491), 8.5 (511), 8.6 (518) | `structures::{beam_modes, flexible_appendage}` | Rust | NR 11, NR 18 (eigen BVP) |
| 10 | 10 Robust Optimal Maneuvers (613–666) | Time-optimal bang-bang control (Pontryagin), robust time- and fuel-optimal profiles, robustified feedforward (input shaping) | 10.1 (613), 10.2 (625), 10.4 (637), 10.5 (652), 10.3 (631) | `control::{time_optimal, input_shaping}` | Rust | rank 7 |
| 11 | 11 Control Moment Gyros (667–740) | Singularity analysis (Binet–Cauchy, singular surfaces), singularity-robust and escape steering, agile multitarget pointing, variable-speed CMGs | 11.2 (669), 11.3 (673), 11.7 (699), 11.8 (706), 11.9 (723), 11.10 (738) | `actuators::cmg` | Rust | NR 2.6 |
| 12 | 12 Solar-Sail Dynamics and Control (741–846) | Solar-radiation-pressure models (12.3) are reusable as an orbit perturbation; the rest is sail-specific | 12.3 (749), 12.13 (807) | `astrodynamics::perturbations::srp` only | Rust | rank 4 |
| 13 | 13 Solar-Sail Missions for Asteroid Deflection (847–900) | Gravity tractor and hovering dynamics; read for the CR3BP applications | 13.3 (868), 13.6 (887) | none | — | read-only |
| 14 | 14 Space Solar Power Satellites (901–934) | Attitude of large inertially oriented structures; read for the gravity-gradient and structural-control examples | 14.4 (915), 14.6 (921) | none | — | read-only |

## 3. What the ranks build, in order

- **Ranks 1–2** are the rotational half of a 6-DOF simulator: state on SO(3),
  Euler's equations, an inertia type, and the closed-form torque-free
  solutions that let every propagator be tested against an exact answer
  rather than against another numerical run.
- **Rank 3** closes the loop: a quaternion-feedback controller with a proof,
  wheels and thrusters as actuators, and allocation. With ranks 1–2 and the
  NR integrators this is the attitude-control demo.
- **Ranks 4–5** are the translational half and the maneuvers a vehicle
  actually flies: Kepler, elements, perturbations, Hohmann, rendezvous, halo.
  Clohessy–Wiltshire is the docking and refueling model.
- **Ranks 6–7** are the analysis substrate: linearize any of the above,
  read its spectrum, discretize it for a flight computer, design an LQR.
- **Rank 8** is the launch-vehicle chapter: TVC with slosh is the model a
  booster's attitude loop is designed against, and it ties directly to
  Sutton chapter 18.
- **Ranks 9–11** add flexibility, optimal maneuvers, and CMGs.
- **Ranks 12–14** are read-only except for the SRP model.

## 4. Coding briefs (one per top-ranked module)

Each brief is the physics statement an agent starts from, the named objects,
the tests that follow from the mathematics, and the language. Equation
numbers are deliberately not quoted here; the agent reads the section and
records the equation tags in the citation sidecar once the reconciled OCR is
available.

### 4.1 `attitude::kinematics` (chapter 5) — Rust

- **Physics.** Attitude is a point on SO(3). A rotation is represented by a
  direction-cosine matrix C ∈ SO(3), by Euler angle sequences, by an eigenaxis
  and angle (Euler's theorem), or by a unit quaternion q ∈ S³ double-covering
  SO(3). The kinematic differential equation relates q̇ (or Ċ) to the body
  angular velocity ω.
- **Named objects.** `DirectionCosineMatrix`, `EulerAngles<Sequence>` with
  the sequence as a type parameter (never a runtime string), `EigenAxisAngle`,
  the library `Quaternion` (story 08), `WieQuaternionAdapter` (scalar-last in,
  scalar-first out, sign convention recorded), `AngularVelocity` in the body
  frame with the frame in the type.
- **Tests.** C(q) = C(−q); round trips q → C → q up to sign; C(q₂ ⊗ q₁) =
  C(q₂) C(q₁) in the library's composition order; the kinematic equation
  integrated with a fixed ω over t returns the eigenaxis rotation by |ω| t;
  ‖q‖ stays 1 to machine precision when integrated with the NR DOPRI5 port
  plus renormalization; every Euler sequence's singularity is at the angle the
  book states.
- **Depends on.** story 08 kernels; `cosmos_numerical::ode`.

### 4.2 `attitude::rigid_body` + `attitude::inertia` + `attitude::torque_free_oracle` (chapter 6) — Rust, CUDA later

- **Physics.** Angular momentum H = J ω about the center of mass; Euler's
  equations J ω̇ + ω × (J ω) = M in the body frame; the inertia matrix J is
  symmetric positive definite, so it has principal axes (the symmetric
  eigenproblem, NR 11). Torque-free motion conserves H and rotational kinetic
  energy; for an axisymmetric body the solution is closed-form (a coning
  motion at a rate set by the inertia ratio), for a general body it is
  given by Jacobi elliptic functions (NR 6.11). Spin about the major or minor
  axis is stable, about the intermediate axis unstable. In a circular orbit
  the gravity-gradient torque 3n² ô × (J ô) appears.
- **Named objects.** `InertiaMatrix` (SPD, with `principal_axes()` returning
  a rotation and three moments), `RigidBodyState { q, ω }`, `EulerEquations`
  as the right-hand side for the NR integrators, `GravityGradientTorque`,
  `AxisymmetricTorqueFreeSolution`, `GeneralTorqueFreeSolution` (elliptic).
- **Tests.** H and energy conserved to the integrator's order over long
  runs; propagator matches the axisymmetric closed form to tolerance; matches
  the elliptic-function solution for a triaxial body; intermediate-axis
  flip reproduced; principal axes are orthonormal and diagonalize J;
  gravity-gradient equilibrium and its stability conditions reproduced.
- **CUDA.** Batched propagation of N rigid bodies with dispersed inertia and
  initial rates is the first attitude Monte Carlo kernel; the Rust propagator
  is the reference.

### 4.3 `attitude::control` (chapter 7) — Rust

- **Physics.** Quaternion-error feedback u = −K_p q_e,vec − K_d ω with a
  Lyapunov function built from the error quaternion and kinetic energy;
  reaction wheels exchange momentum with the body (total H conserved);
  thruster selection is a linear program over on-times; pulse-width
  pulse-frequency modulation turns a continuous command into on-off pulses.
- **Named objects.** `QuaternionFeedbackGains`, `ReactionWheelSet`,
  `MomentumManagement`, `JetSelection` (LP from NR 10.10), `PwpfModulator`
  with its four named constants, `CmgSteering` (pseudo-inverse via SVD).
- **Tests.** Convergence from random initial attitudes; total momentum
  conserved with wheels; LP allocation is feasible and minimum-fuel on the
  book's thruster layout; PWPF limit cycle matches the describing-function
  prediction of chapter 9.

### 4.4 `astrodynamics::{kepler, elements, rendezvous}` (chapters 3–4) — Rust

- **Physics.** Two-body motion in a conic; Kepler's time equation solved by
  Newton (NR 9.4) for elliptic, parabolic, and hyperbolic cases; classical
  elements ↔ Cartesian state; vis-viva; Hohmann Δv; relative motion in a
  circular target orbit obeys the Clohessy–Wiltshire equations with a
  closed-form state transition matrix.
- **Tests.** Kepler round trips at all eccentricities including near-parabolic;
  element ↔ state round trips; energy and momentum constant along a
  propagated conic; CW closed form matches the nonlinear two-body relative
  propagation to second order in separation; Hohmann Δv identities.
- **Depends on.** `Cosmos/Source/Astrodynamics` C++ (golden vectors for
  two-body and J2), `cosmos_numerical::root_finding`.

### 4.5 `vehicle::slosh_pendulum` + `vehicle::tvc_loop` (section 9.1) — Rust

- **Physics.** A booster with liquid propellant modeled as a rigid body plus
  pendulum (or spring-mass) slosh modes; the TVC gimbal produces a moment;
  the attitude loop must stabilize the rigid mode while not destabilizing
  the slosh mode, which depends on where the slosh mass sits relative to the
  center of mass.
- **Named objects.** `SloshMode { mass, length, attachment }`,
  `BoosterPlant` (linearized state space), `TvcGimbal` (limits from Sutton
  chapter 18), `AttitudeLoop` with gains and a filter.
- **Tests.** Slosh eigenfrequency equals the pendulum formula; the book's
  stability boundary (slosh mass above vs below the center of mass)
  reproduced from the root locus; gain and phase margins computed from the
  chapter 2 frequency-domain tools.

### 4.6 `control::{discretize, lqr}` (chapter 2) — Rust

- **Physics.** A continuous plant ẋ = A x + B u sampled with zero-order hold
  becomes x_{k+1} = e^{AT} x_k + (∫₀ᵀ e^{Aτ} dτ) B u_k; the LQR gain comes
  from the algebraic Riccati equation.
- **Tests.** Discretization of a double integrator matches the analytic
  matrices; LQR for a scalar plant matches the closed form; the discrete
  closed loop is stable when the continuous one is designed stable and the
  sample rate respects the book's guidance.

## 5. Reading order for the first month

1. §5.4 with the quaternion lab open; write the adapter and its double-cover
   tests. 2. §5.5 kinematic equations; integrate with the DOPRI5 port.
3. §6.1–6.4 Euler's equations and inertia; §6.5 the axisymmetric oracle.
4. §6.7 stability; §6.10 gravity gradient. 5. §7.3 quaternion feedback.
6. §3.4–3.5 Kepler and elements. 7. §4.6 rendezvous. 8. §1.3 and §2.4–2.5.
9. §9.1 TVC with slosh. Each step ends with the seven artifacts.
