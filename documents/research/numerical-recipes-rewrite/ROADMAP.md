# Numerical Recipes 3e — chapter roadmap ordered by relevance to spacecraft simulation and GNC

**Status:** working plan, 2026-09-01. Supersedes nothing; extends the reading
guide at `NumericalRecipes-3e/READING_GUIDE.md` (five arcs, pipeline view) with
a book view: every chapter ranked, with the physics that makes it relevant, the
module it becomes, and the language it is written in.
**Navigation:** PDF page = printed page + 24, constant across the book.
**Relevance signal:** the dated capability record in
[`spacex-signal-2026-09-01.md`](spacex-signal-2026-09-01.md) plus what already
exists in `Cosmos/Source`. The five phrases that set the order: six-degree-of-
freedom simulation, Monte Carlo dispersions, post-flight correlation, real-time
determinism, Rust alongside C++.

## 1. Language policy

- **Rust is the primary language for every new numerical module.** Home:
  `Cosmos/Rust/` (a Cargo workspace; crate `cosmos_numerical` mirrors
  `Cosmos/Source/Numerical`, later `cosmos_astrodynamics`). The field trait is
  wildrider's `FieldOperations` (`Stunticons/wildrider`), reused by path
  dependency so the mathematical spine has one definition.
- **C++ stays where it is.** `Cosmos/Source` keeps chapter 17 and everything the
  C++ simulation stack consumes (Cosmos executables, the WASM orbit demo, the
  Isaac and Blender pipelines). A C++ twin of a new module is written only when
  that stack needs it. When both languages implement a method, a
  cross-language golden-vector test is mandatory: the C++ build emits vectors,
  the Rust test reads them, and agreement is asserted to a stated tolerance.
  Precedent: the CRC-16 telemetry harness in `anysignal-demo`.
- **CUDA C++ only for data-parallel workloads with a measured CPU baseline:**
  batched Monte Carlo propagation, batched small factorizations, PDE stencils,
  FFTs. The Rust CPU implementation exists first and is the reference the
  kernel is checked against. Existing CUDA lives in `Stunticons/Source` (CG,
  BiCGSTAB solvers) and `CUDACFD/`.
- `f64` by default; generic over the field trait where it costs nothing; no
  `unsafe` in Rust numerics except the FFI boundary to CUDA.
- Style: Rust follows the hand style already in wildrider (two-space indent,
  brace on its own line, spelled-out names, `snake_case` functions,
  `CamelCase` types), matching the C++ conventions in `Cosmos/Source`.

## 2. The first-principles protocol (per section)

Reading a section is not finished until all seven exist. This is the rule
that keeps the rewrite from aping the book.

1. **Physics first.** State the physical system and its governing equations
   in the library's own terms: units, frames, the group the state lives in
   (story 08). Example: an orbit is a Hamiltonian system on T*R³; attitude is
   a curve on SO(3).
2. **Mathematical structure.** Name the property the algorithm exploits.
   Symmetric positive definite ⇒ Cholesky exists and is stable without
   pivoting. Hamiltonian ⇒ a symplectic integrator preserves a modified
   energy. Stiff Jacobian ⇒ implicit stages. Second-order ODE without y' ⇒
   Störmer.
3. **Named objects.** Every noun in the mathematics becomes a type (tableau,
   factor, controller law, tolerance). Every constant becomes a named,
   asserted, injected parameter. Every verb becomes a testable function.
4. **Derive, then compare.** Derive the recurrence from the equations
   yourself, write it, and only then read NR's prose to see what they chose
   and why. Never open `NR_C301/code/*.h` while writing; the code is not
   redistributable and its design is the thing being replaced.
5. **Tests from the mathematics.** Property tests that follow from step 2:
   Butcher order conditions, reconstruction A = LLᵀ to machine precision,
   bounded energy drift, exact conservation by projection, monotone
   convergence with step refinement at the claimed order. Plus golden vectors
   across languages.
6. **Citation sidecar** (story 05): section, printed and PDF page, equation
   tags from `equation_index.json`, and the substitute reference used where
   NR is dated.
7. **Ledger entry** in `READING-LEDGER.md`: what NR got right, what aged,
   which substitute was used, and the module and tests that resulted.

Precedent already in the repo: `documents/StormerRule.tex` (derivation with
physical motivation and HNW order theory) plus
`Cosmos/Source/Astrodynamics/Propagators/NumerovOrbit.h` (architecture and
references in the header). Every new method gets that pair.

## 3. Chapter ranking

Tier 0 is read before anything else: §1.1 (p. 8), §5.7 (p. 229), §22.2
(p. 1163), then §1.3–1.5 (pp. 17–36) read adversarially. About 35 pages.

| Rank | Chapter | Why it matters here (the physics) | Sections that carry it (printed page) | Becomes | Language | Status |
|---|---|---|---|---|---|---|
| 1 | 17 Integration of ODEs | The propagator. 6-DOF translational and rotational equations of motion; orbits are Hamiltonian; stiff chemistry | 17.0 (899), 17.1 (907), 17.2 (910), 17.4 (928), 17.3 (921), 17.5 (931), 17.6 (942), 17.7 (946) | `ode::runge_kutta`, `ode::stormer`, `ode::bulirsch_stoer`, `ode::rosenbrock`, `ode::multistep` | Rust port of the C++; C++ retained | C++ done for RK4/DOPRI5/DOPR853, PI control, dense output, Störmer/Numerov. BS, stiff, multistep not started |
| 2 | 2 Linear algebra | Covariance P = LLᵀ, square-root filters, least-squares normal equations, mass-matrix solves in multi-body dynamics, network solves | 2.9 Cholesky (100), 2.10 QR (102), 2.6 SVD (65), 2.3 LU (48), 2.4 tridiagonal/banded (56), 2.7 sparse (75), 2.5 iterative improvement (61) | `linear_algebra::{cholesky, qr, svd, lu, banded}` as factorization types owning their factors | Rust first; CUDA later for batched small Cholesky | Not started in Cosmos; CUDA CG/BiCGSTAB exist in Stunticons |
| 3 | 7 Random numbers | Monte Carlo dispersion analysis: 3σ initial conditions, aero, thrust, mass; reproducible streams keyed by run id | 7.1 uniform (341), 7.3 other deviates (361), 7.4 multivariate normal (378), 7.8 Sobol (403), 7.7 simple MC (397), 7.9 adaptive MC (410) | `random::{generator, deviates, multivariate_normal, sobol}` | Rust; CUDA for batched propagation (the canonical data-parallel workload) | Not started |
| 4 | 15 Modeling of data | Post-flight correlation, orbit determination, parameter identification (aero coefficients, thrust curves), covariance of estimates | 15.4 linear LS (788), 15.5 Levenberg–Marquardt (799), 15.6 confidence limits (807), 15.7 robust (818), 15.8 MCMC (824), 15.9 GP regression (836) | `estimation::{least_squares, levenberg_marquardt, confidence}` built on rank-2 factorizations | Rust | Not started |
| 5 | 9 Root finding | Kepler's equation, targeting, trim, the inner solve of every implicit stage | 9.1 (445), 9.3 Brent (454), 9.4 Newton (456), 9.6 Newton systems (473), 9.7 globally convergent (477) | `root_finding::{bisection, brent, newton, line_search, globally_convergent_newton}` with the line-search policy separated | Rust | Not started |
| 6 | 3 Interpolation | Aero databases CL, CD(Mach, α), atmosphere tables, engine maps, dense output | 3.1 lookup (114), 3.2 polynomial (118), 3.3 cubic spline (120), 3.4 rational (124), 3.6 grids in multi-D (132), 3.7 scattered (139) | `interpolation::{lookup, polynomial, cubic_spline, grid}` | Rust; C++ `Lookup.h` exists | Lookup and linear 1-D done in C++ |
| 7 | 10 Minimization | Control allocation over actuators (LP/QP), trim, outer loop of trajectory optimization | 10.1–10.3 (490–496), 10.5 simplex (502), 10.7 Powell (509), 10.8 CG (515), 10.9 BFGS (521), 10.10 LP simplex (526), 10.11 interior point (537) | `optimization::{bracketing, brent, bfgs, conjugate_gradient, linear_program}` | Rust | Bracketing, parabolic, golden section in C++; `mins.h` still NR-shaped |
| 8 | 18 Two-point BVPs | Boost-back and landing-burn targeting; substitute Betts for collocation | 18.1 shooting (959), 18.2 fitting point (962), 18.3 relaxation (964), 18.5 mesh allocation (981) | `boundary_value::{shooting, multiple_shooting}` | Rust | Not started |
| 9 | 11 Eigensystems | Principal axes of inertia; linearized stability of a controller (spectrum of ∂f/∂x); structural and POGO modes | 11.1 Jacobi (570), 11.3 tridiagonal reduction (578), 11.4 tridiagonal eigen (583), 11.6 nonsymmetric (590), 11.7 QR on Hessenberg (596) | `linear_algebra::eigen::{symmetric, nonsymmetric}` | Rust | Not started |
| 10 | 14 Statistical description | Flight-versus-simulation correlation: are two distributions the same; Monte Carlo result statistics; telemetry smoothing | 14.1 moments (721), 14.2 (726), 14.3 K-S tests (730), 14.9 Savitzky–Golay (766) | `statistics::{moments, distribution_tests, savitzky_golay}` | Rust | Not started |
| 11 | 12 + 13 FFT and spectral | Combustion instability, POGO, IMU conditioning, unevenly sampled telemetry | 12.2 FFT (608), 12.3 real FFT (617), 13.4 power spectrum (652), 13.5 digital filtering (667), 13.8 Lomb–Scargle (685), 13.3 Wiener (649) | `spectral::{fft, power_spectrum, filters, lomb_scargle}` | Rust; CUDA for batched FFT | Not started |
| 12 | 4 Quadrature | Δv and impulse integrals, ballistic coefficients, the basis of pseudospectral collocation | 4.6 Gaussian (179), 4.3 Romberg (166), 4.7 adaptive (194), 4.5 variable transformation (172) | `quadrature::{gauss, romberg, adaptive}` keeping NR's quadrature abstraction, renamed | Rust | Not started |
| 13 | 5 Evaluation of functions | Finite-difference Jacobians (EKF linearization); Chebyshev fits (JPL ephemerides are Chebyshev series; thermodynamic property fits); Clenshaw recurrences (Legendre for gravity fields) | 5.7 derivatives (229), 5.8 Chebyshev (233), 5.4 recurrences and Clenshaw (219), 5.1 polynomials (201) | `functions::{numerical_derivative, chebyshev, clenshaw}` | Rust | Not started |
| 14 | 6 Special functions | Spherical harmonics for the geopotential (J2…Jn, EGM); Bessel functions for cylindrical acoustic modes of a chamber; Jacobi elliptic functions give the exact torque-free rigid-body solution used to test attitude propagators | 6.7 spherical harmonics (292), 6.5 Bessel (274), 6.11 elliptic (309) | `special_functions::{legendre, spherical_harmonics, bessel, elliptic}` | Rust | Not started; NR accuracy caveat: verify beyond 6 digits against a reference |
| 15 | 20 PDEs | One-dimensional nozzle Euler equations, regenerative-cooling and thermal-soak diffusion, elliptic pressure solves | 20.1 flux-conservative (1031), 20.2 diffusive (1043), 20.3 multi-D (1049), 20.6 multigrid (1066), 20.7 spectral (1083) | `pde::{finite_volume, diffusion, multigrid}` | CUDA C++ primary with a Rust reference; ties to `CUDACFD/` | Toy convection solvers exist in `CUDACFD/` |
| 16 | 21 Computational geometry | Quaternions and rotation composition (done); KD trees for star-catalog lookup in a star tracker; Delaunay for meshes | 21.5 spheres and rotations (1128), 21.2 KD trees (1101), 21.6 Delaunay (1131) | `geometry::kd_tree`; rotations stay in the quaternion lab | Rust | 21.5 done in `Cosmos/QuaternionConventionLab` (C++ and Rust) |
| 17 | 19 Integral equations and inverse theory | Inverse heat conduction for chamber-wall heat flux, regularized system identification | 19.4 inverse problems (1001), 19.5 linear regularization (1006) | `inverse::regularization` | Rust | Not started |
| 18 | 22 Less-numerical | Machine epsilon on the target (Jetson, TX2i); CRC for telemetry integrity; Gray codes for encoders; arbitrary precision to generate reference solutions for tests | 22.2 (1163), 22.4 CRC (1168), 22.3 Gray (1166), 22.6 arbitrary precision (1185) | `utilities::{machine_parameters, crc}`; high-precision references for golden vectors | Rust | CRC-16 exists in the telemetry harness |
| 19 | 1 Preliminaries | Error, accuracy, stability vocabulary; the 2007 C++ philosophy to read against | 1.1 (8), 1.3–1.5 (17–36) | no module | — | Tier 0 |
| 20 | 8 Sorting and selection | Indexing and ranking of telemetry; standard libraries cover the rest | 8.4 indexing and ranking (428) | none; use `sort_by` | — | Read only |
| 21 | 16 Classification and inference | Mode detection in telemetry (HMM, Viterbi); the rest is superseded by modern ML | 16.3 Viterbi (850), 16.4 HMM (856) | none for now | — | Read only |

## 4. What each rank means for the simulation we are building

- Ranks 1–3 give a dispersed 6-DOF simulation: propagate (17), draw and
  transform uncertainties (7 with 2.9), and solve the linear systems inside
  every step (2).
- Ranks 4, 9, 10 give post-flight correlation and stability analysis: fit
  models to flight data (15), compare distributions (14), and read the
  spectrum of the linearized dynamics (11).
- Ranks 5, 7, 8 give guidance and targeting: root finding (9), allocation and
  optimization (10), and boundary-value targeting (18).
- Ranks 6, 12, 13, 14 give the physics tables and force models a real vehicle
  needs: aero and atmosphere interpolation (3), impulse integrals (4),
  Chebyshev ephemerides and Legendre gravity fields (5, 6).
- Ranks 11, 15 reach into propulsion: spectral analysis of combustion
  instability (12–13) and the nozzle and cooling PDEs (20).
- Ranks 16–21 are utilities and reading.

Where NR is dated the substitute is fixed now, not later: Hairer–Nørsett–Wanner
and Hairer–Lubich–Wanner for 17; Trefethen–Bau and Higham for 2 and 11;
Markley–Crassidis and Tapley–Schutz–Born for the estimators NR does not contain;
Betts for 18; Montenbruck–Gill for force models.

## 5. Sequence for the next ten working sessions

1. Rust workspace at `Cosmos/Rust/`; port the PI step-size controller with a
   golden-vector test against `ComputePIStepSize.h`. Proves the twin
   discipline on the smallest object that has one.
2. Cholesky as a factorization type (2.9) from first principles, with the
   reconstruction and non-positive-definite tests, and the covariance
   square-root use case.
3. Multivariate normal deviates (7.4) on top of 2; a reproducible generator
   (7.1) keyed by run id.
4. Port the DOPRI5 tableau and step (17.1–17.2) to Rust with Butcher
   order-condition tests; golden vectors from the C++ tests.
5. QR (2.10) and linear least squares (15.4); then Levenberg–Marquardt (15.5).
6. Newton and globally convergent Newton (9.6–9.7) with the line search
   separated; Kepler's equation as the worked example.
7. Cubic spline and gridded interpolation (3.3, 3.6); an aero-table type.
8. Symmetric eigensolver (11.1, 11.3–11.4); inertia principal axes and
   linearized stability.
9. Störmer/Numerov port and the torque-free rigid body via Jacobi elliptic
   functions (17.4, 6.11) as the attitude-propagator test oracle.
10. Batched Monte Carlo propagation in CUDA against the Rust reference from 1–4.

Each session produces the seven artifacts of §2 for its sections.
