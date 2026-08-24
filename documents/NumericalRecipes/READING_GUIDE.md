# Numerical Recipes → Propulsion, Spacecraft Simulation, and GNC

**Start here. Do not read Numerical Recipes cover to cover.**

NR3 is a 1235-page cookbook. The C++ in `NR_C301/code` is C with `struct` and a custom vector class. The useful thing in the book is the *judgment* — which method to use, what fails, and the implementation details that never made it into the prose (FSAL, dense-output polynomials, scaled RMS error, Rosenbrock LU reuse, Störmer as a second-order method). Those details live in the headers. We already started extracting them the right way in `Cosmos/Source/Numerical/ODE/RKMethods/`. This guide says what to read next, in what order, and which books/papers replace NR when NR is wrong or silent.

Companion parse (do not re-parse the PDF):

`workspace2/Data/Public/books/Physics/Press W.H., Teukolsky S.A., Vetterling W.T., Flannery B.P. - Numerical Recipes The Art of Scientific Computing - 2007/parsed/`

---

## 0. What you should start reading

Forced rank. This is a 6DOF / GNC / propulsion sequence, not a numerical-analysis PhD.

### This week — the numerical core of a spacecraft sim

| # | Read | Why | Hours |
|---|------|-----|-------|
| 1 | **NR §17.0** (book p. 899) | Driver / stepper / algorithm / Output split. Their *only* architectural idea worth keeping. Also their (dated) ranking of RK vs Bulirsch-Stoer vs multistep. | 45 min |
| 2 | **NR §17.1–17.2** with `NR_C301/code/{odeint,stepper,stepperdopr5}.h` **side-by-side** with `Cosmos/Source/Numerical/ODE/RKMethods/` | Adaptive RK is the production workhorse for 6DOF with events. Principles are buried in `StepperDopr5::dy`, `Controller::success`, `dense_out`. You already rewrote this properly. Read to extract what is still missing (event location on dense output, component-wise tolerances). | 3 h |
| 3 | **Hairer, Nørsett, Wanner vol. I** §§II.4–II.6 (you own this) | The audit standard for `RKMethods`. Butcher tableaus, embedded pairs, FSAL, local extrapolation. NR is a tourist map; Hairer is the terrain. | 2–3 h |
| 4 | **NR §9.0–9.3** + `roots.h` | Event detection: eclipse, thruster on/off, equator crossing, RPO range gate. Bisection → Brent. You were already screened on this. | 1.5 h |
| 5 | **NR §17.4** + your `StormerRule.tex` + `StormerMethods/` | Second-order conservative systems \(y''=f(x,y)\). Orbit and rigid-body *kinematics-as-second-order* live here. NR's Störmer-Verlet/Bulirsch path. | 1.5 h |

Do **not** open chapters 6, 8, 12, 16, 22. Do **not** start rewriting all 162 headers.

### Next — the two things NR gets wrong for spacecraft

| # | Read | Why |
|---|------|-----|
| 6 | **NR §17.5** stiff ODEs + `stepperross.h` / `steppersie.h` | Thermal, power, chemical kinetics, RCS pulses. Rosenbrock is the NR answer. Production answer is BDF/CVODE or IMEX. |
| 7 | **NR §17.6** (no code — they refused to implement it) then **Berry & Healy 2004** | Gauss-Jackson / summed Adams is what space-surveillance and a lot of special-perturbations catalogs actually run. NR calls predictor-corrector “not given in this book.” That sentence is why you cannot learn operational orbit propagation from NR. |
| 8 | **NR §2.3, 2.6, 2.7** LU / SVD / sparse + `ludcmp.h` `svd.h` `linbcg.h` | Never invert. Factor/solve. Conditioning. Sparse thermal networks. Your BiCGSTAB in Stunticons already occupies this slot. |

### Then — physics books you already own, not more NR

| # | Book (local) | Use |
|---|--------------|-----|
| 9 | Curtis, *Orbital Mechanics for Engineering Students* | Two-body, Lambert, CW. Fast refresh, not the reference. |
| 10 | Prussing & Conway, *Orbital Mechanics* **or** Bate, Mueller, White | Hohmann, transfers, primer-vector taste. |
| 11 | Sidi, *Spacecraft Dynamics and Control* | Practical ADCS. Read attitude kinematics + gravity-gradient + wheels first. |
| 12 | Wie, *Space Vehicle Dynamics and Control* | When Sidi is too cookbook and you want the Lyapunov/ω-equation treatment. |
| 13 | Sutton & Biblarz; Humble *Space Propulsion Analysis and Design* | Rocket equation, nozzles, blowdown. NR will not teach you this. |

### Acquire next (you do not have these locally)

Ranked by how soon a spacecraft-sim job will make you need them:

1. **Montenbruck & Gill, *Satellite Orbits*** — *the* numerical-methods book for operational orbit. Force models, variational equations, integrators.
2. **Hairer & Wanner vol. II (stiff/DAE)** — NR §17.5 is a pamphlet; this is the book. You have vol. I and *Geometric Numerical Integration*; you are missing vol. II.
3. **Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*** — attitude estimation from first principles (QUEST, q-method, MEKF). Sidi/Wie do not replace this.
4. **Schaub & Junkins, *Analytical Mechanics of Space Systems*** — MRP/quaternion conventions, coupled orbit-attitude. Closest textbook to Basilisk's dynamics.
5. **Vallado, *Fundamentals of Astrodynamics and Applications*** (4e/5e) — the desk reference. Algorithms, frames, time, drag, third body.
6. **Zipfel, *Modeling and Simulation of Aerospace Vehicle Dynamics*** — how a 6DOF *program* is structured (frames, aerodynamics tables, multi-rate). Basilisk paper first if you only have time for one architecture doc.

---

## 1. How NR maps onto a spacecraft simulator

A 6DOF spacecraft sim is not “an ODE.” It is several IVPs, discrete events, table lookups, and linear solves, coupled on a clock.

```
clock / orchestrator
        │
        ├─ translational IVP     r̈ = a_grav + a_drag + a_srp + F_thrust/m     NR 17, 17.4
        ├─ rotational IVP        Iω̇ + ω×Iω = τ                               NR 17; quats are not in NR
        ├─ event functions       g(t,y)=0  (eclipse, thrust, eclipse-exit)     NR 9 + dense output 17.2
        ├─ table / atmosphere    ρ(h), Cd(M,α), ephemeris                      NR 3
        ├─ linear algebra        Kalman, least squares, implicit step          NR 2
        ├─ UQ / MC               dispersions, seed, correlations               NR 7
        ├─ parameter ID / OD     batch LS, EKF linearization                   NR 15, 2.6
        └─ (propulsion CFD)      NS + stiff chemistry                          NR 20 + 17.5
```

NR has **no** spacecraft content: no frames, no quaternions, no Kepler, no STM, no SRP, no gravity harmonics. Use it as the numerical *substrate*, then go to Curtis/Vallado/Montenbruck/Sidi/Markley for the physics.

---

## 2. NR C++ vs how we actually write C++

This is the rewrite contract. Principles are buried in NR *code*, not in NR *types*.

### What NR3 actually is

`nr3.h` is 2007-era C++98:

- `using namespace std;`
- a macro that redefines `throw`
- `NRvector` / `NRmatrix` instead of `std::vector` (or a proper `NVector<N,Field>`)
- typedefs `Doub`, `Int`, `VecDoub`
- public `struct` with raw loops
- stepper holds **references** to `x, y, dydx` owned by `Odeint` — lifetime as a language
- Butcher coefficients are `static const Doub` locals inside `StepperDopr5::dy`
- error control, step, dense output, and the method are one class

That last point is the one NR *got conceptually right* in §17.0.1 (driver / stepper / algorithm / Output) and then **failed to express in types**.

### What we already do (keep doing this)

| Concern | NR | Cosmos / Stunticons |
|---------|----|---------------------|
| Vector | `NRvector` / `VecDoub` | `Algebra::Modules::Vectors::NVector<N,Field>` |
| RK stages | unrolled `k2…k6` in `dy()` | `Coefficients::KCoefficients<S,ContainerT>` |
| Tableau | magic numbers in `dy()` | `ACoefficients`, `BCoefficients`, `CCoefficients`, `DOPRI5Coefficients` |
| One step | `StepperDopr5::dy` + `step` + `Controller` in one type | `CalculateNewYAndError`, `CalculateScaledError`, `ComputePIStepSize`, `StepWithPIControl` |
| Integrate | `Odeint<Stepper>::integrate` | `IntegrateWithPIControl` / `HigherOrderIntegrateWithPIControl` |
| Dense output | `dense_out` polynomial inside the stepper | `CalculateDenseOutputCoefficient`, `calculate_hermite_interpolation` |
| Störmer | `stepperstoerm.h` | `StormerMethods/NumerovStep.h` + `StormerRule.tex` |
| Sparse solve | `linbcg.h` | `Algebra::Solvers::BiconjugateGradientStabilized` (CUDA) |
| Tests | none in `NR_C301` | `Cosmos/Source/UnitTests`, GoogleTest, `-Wall -Wextra -pedantic -Werror`, C++20 |
| Namespaces | global | `Numerical::ODE::RKMethods`, `Algebra::…` |

There is a **transitional** NR-shaped stack still in tree: `ODE/StepperDopr5.h`, `StepperBase.h`, `ODEInt.h`, with a comment citing `github.com/blackstonep/Numerical-Recipes`. That is the thing we do **not** extend. New work goes in `RKMethods/` (and eventually a sibling `Roots/`, `LinearAlgebra/`, `Interpolation/` at the same granularity).

`T1000/Cosmos/AUDIT.md` already records that `CalculateNewYAndError` is the production path and `CalculateNextStep` has an off-by-one. Hairer I is the correctness reference, not NR.

### Rewrite rules when a “principle is buried in the code”

1. **Promote numbers to types.** If `dy()` contains `a21=0.2, c2=0.2, e1=71/57600`, those are `ACoefficients` / `CCoefficients` / `DeltaCoefficients`. Adding DOP853 then means a new coefficients file, not a new 400-line stepper.
2. **One class, one invariant.** Computing \(k_i\) is not choosing \(h\). Choosing \(h\) is not writing output. Event localization is not the stepper.
3. **No reference members to the caller's state.** Pass `StepInputs` / `IntegrationInputs` by value or span. NR's `StepperBase(VecDoub &y, Doub &x)` is a use-after-scope waiting to happen.
4. **Callables, not inheritance, for the RHS.** NR uses `template<class D>` plus a `typedef D Dtype` convention. We take `DerivativeType&&` and `std::forward`. Do not put `std::function` on the millions-of-calls path.
5. **Do not copy NR headers.** The license is not a production license. Re-derive from Hairer / original papers and write against our Algebra types.
6. **Test against analytic problems first.** SHO, two-body specific energy, Van der Pol (already in `RhsVanDerPol.h`), then a stiff linear test, then a spacecraft RHS.

---

## 3. NR chapter map (only the useful subset)

Priority: **P0** = read now, **P1** = this month, **P2** = when the job/task demands it, **skip** = not for this track.

### P0 — you cannot ship a sim without these

**§1.1 Error, accuracy, stability** (p. 8). Floating-point, cancellation, what “tolerance” means. Then impose *component-wise* `atol + rtol*|y|` (NR §17.2 does this; a single global `eps` is wrong for meters vs quaternion vs kelvin vs kg).

**§9 Root finding** (p. 442) — `roots.h`
- 9.1 bracketing / bisection
- 9.3 Brent (production 1D root)
- Skip 9.5 polynomials until Lambert/Kepler implementations need it
- 9.6–9.7 Newton for systems: implicit residual, not the first 1D event locator

Application: `g(t)=n·r` (eclipse), `h-h_gate` (altitude), `range - R_cmd`, thruster pulse edges. Detect sign change on the accepted step, **localize on dense output**, cut, apply the discrete jump once, restart. NR never writes that paragraph; it is the whole point of `dense_out` in a vehicle sim.

**§17 ODEs** (p. 899) — the chapter
| Section | Code | Spacecraft / propulsion use |
|---------|------|-----------------------------|
| 17.0 | `odeint.h` `stepper.h` | Architecture. Steal the layering, not the types. |
| 17.1 | `rk4.h` | Pedagogy / interviews only. |
| 17.2 | `stepperdopr5.h` | **Default 6DOF stepper.** DOPRI5, PI controller, dense output. |
| webnote 20 | `stepperdopr853.h` | High-accuracy coast arcs. You already have `DOPR853Coefficients`. |
| 17.3 | `stepperbs.h` | Smooth, expensive \(f\). Rarely the 6DOF default (events kill extrapolation). |
| 17.4 | `stepperstoerm.h` | \(y''=f(y)\) orbits; symplectic-adjacent. Your Numerov/Störmer work. |
| 17.5 | `stepperross.h` `steppersie.h` | Stiff: chemistry, thermal RC networks, fast electrical. |
| 17.6 | *no code* | Multistep. Read the prejudices, then go to Berry-Healy / Hairer III. |
| 17.7 | `stochsim.h` | Gillespie. Optional; combustion CFD uses CVODE, not SSA, at engineering scale. |

**§3 Interpolation** (p. 110) — `interp_1d.h` `interp_linear.h`
Atmosphere, aero tables, thrust vs time, ephemeris samples. Cubic spline is the default; do not polynomial-extrapolate off a table. Scattered-data (§3.7 RBF/kriging) is later (gravity anomalies, on-orbit residual maps).

### P1 — you will hit these in the first months of a sim job

**§2 Linear algebra** (p. 37)
- 2.3 LU: implicit RK / Rosenbrock / Newton
- 2.4 tridiagonal: 1D thermal, spline moments
- 2.6 SVD: rank-deficient geometry, batch OD, Wahba-adjacent
- 2.7 sparse + BiCG: 3D thermal, FEM
- 2.9 Cholesky: covariance, attitude-error Fisher
- **Never** `gaussj` a system you could factor. NR says this; people still invert `FIM` in Kalman derivations.

**§7 Random numbers** (p. 340) — `ran.h` `deviates.h` `multinormaldev.h`
Monte Carlo dispersions. Seed, independent streams per axis, no `rand()`. Multivariate normal for correlated navigation errors. Sobol (§7.8) if you do UQ properly.

**§15 Modeling of data** (p. 773)
Linear LS, SVD fit, Levenberg-Marquardt. This is batch orbit determination and actuator calibration, not “statistics class.” MCMC (§15.8) is optional; EKF/UKF are not in NR — go to Crassidis/Markley.

**§10 Optimization** (p. 487)
1D Brent for line search / TOF. Quasi-Newton later. Skip simplex/annealing as production guidance. Trajectory optimization is **Betts** or primer-vector literature, not NR amoeba.

**§18 BVP / shooting** (p. 955)
Lambert is a BVP. Simple shooting for two-point targeting. Operational Lambert is not NR `shoot.h`. Still worth one afternoon so you recognize the pattern.

### P2 — propulsion, CFD, estimation extras

**§20 PDEs** (p. 1024) — flux-conservative IVP, diffusion, spectral
This is the CFD chapter. You already have Stunticons convection / finite difference / LBM-adjacent CUDA. NR §20.1 is a first-year conservative-form lecture, not Toro. After NR 20.0–20.2, go to Toro *Riemann Solvers* or LeVeque *FVM* (acquire) and to the papers on stiff chemistry (SUNDIALS / pyJac).

**§11 Eigensystems** — inertia principal axes, linearization modal analysis, STM eigenvalues (stability of relative motion).

**§4 Quadrature** — gravitational harmonics, line-of-sight integrals, rare.

**§5.7 Numerical derivatives** — `dfridr.h` for finite-difference Jacobians when you do not have analytic ∂f/∂y. Production GNC wants analytic or automatic differentiation.

**§13 FFT / PSD** — you already have `CombustionInstability/Source/SpectralAnalysis`. NR 13.4 is the textbook PSD.

**§21.5 Spheres and rotations** (p. 1128)
The *only* NR section that touches SO(3). It is computational geometry, not spacecraft attitude. Read Markley or Schaub instead. KD-trees (§21.2) matter for conjunction / RPO gating.

### Skip for this track

Ch. 6 special functions (except erf when doing Gaussian nav errors — `std::erf`). Ch. 8 sorting (`std::ranges`). Ch. 12 FFT implementation (use a library). Ch. 14 cookbook stats. Ch. 16 ML. Ch. 19 integral equations unless you are doing tomography. Ch. 22 CRC — you already know CRC from telemetry harness work; NR `icrc.h` is not a reason to read 22.

---

## 4. Companion books — owned vs get

Paths are under `workspace2/Data/Public/books/EngineeringPhysics/` unless noted.

### Numerical methods (the stack above NR)

| Book | Status | Role |
|------|--------|------|
| Hairer, Nørsett, Wanner — *Solving ODEs I, Nonstiff* (2ed 2008) | **owned** | RK, dense output, PI step-size. Cosmos audit reference. |
| Hairer, Lubich, Wanner — *Geometric Numerical Integration* | **owned** | Symplectic / Störmer-Verlet / Lie-group integrators. Long-term orbit, attitude on SO(3). |
| Hairer & Wanner — *Solving ODEs II, Stiff and DAE* | **GET** | Missing. Chemistry, thermal, Rosenbrock *theory*, BDF, index-1 DAE (constraints). |
| Deuflhard & Bornemann — *Scientific Computing with ODEs* | optional | More modern driver/stepper talk. |
| Hairer already beats NR on every ODE topic except “which C++ file.” |

### Astrodynamics / GNC

| Book | Status | Role |
|------|--------|------|
| Curtis — *Orbital Mechanics for Engineering Students* | **owned** (two copies) | Pedagogy. |
| Prussing & Conway — *Orbital Mechanics* | **owned** | Transfers, primer-vector on-ramp. |
| Bate, Mueller, White — *Fundamentals of Astrodynamics* | **owned** | Classic. |
| Hintz — *Orbital Mechanics and Astrodynamics* | **owned** | Extra problems. |
| Sidi — *Spacecraft Dynamics and Control* | **owned** | Practical ADCS. |
| Wie — *Space Vehicle Dynamics and Control* | **owned** | Nonlinear/robust attitude. |
| Montenbruck & Gill — *Satellite Orbits* | **GET** | Force models + numerical integration + OD. First acquisition. |
| Vallado — *Fundamentals of Astrodynamics and Applications* | **GET** | Desk reference (frames, time, algorithms). |
| Battin — *An Introduction to the Mathematics and Methods of Astrodynamics* | GET later | Lambert, universal variables, the book people cite. |
| Markley & Crassidis — *FSADC* | **GET** | Attitude determination. MATLAB examples on the Buffalo site. |
| Schaub & Junkins — *Analytical Mechanics of Space Systems* | **GET** | Coupled 6DOF, MRPs, Basilisk's intellectual parent. |
| Tapley, Schutz, Born — *Statistical Orbit Determination* | GET when you own an OD filter | Batch/sequential OD. |
| Wertz (ed.) — *Spacecraft Attitude Determination and Control* | optional, historical | The 1978 encyclopedia. Markley supersedes the estimation parts. |
| Fehse — *Automated Rendezvous and Docking of Spacecraft* | GET for RPO | Closing, hold points, CW, sensors. |
| Zipfel — *Modeling and Simulation of Aerospace Vehicle Dynamics* | GET for sim architecture | 6DOF program structure. Read the Basilisk papers first. |

### Propulsion / combustion (you are already well-stocked)

| Book | Status | Role |
|------|--------|------|
| Sutton & Biblarz — *Rocket Propulsion Elements* | **owned** | Elements. |
| Humble — *Space Propulsion Analysis and Design* | **owned** | Vehicle-level. |
| Forman Williams — *Combustion Theory* | **owned** | |
| Poinsot & Veynante — *Theoretical and Numerical Combustion* | **owned** | |
| Norbert Peters — *Turbulent Combustion* | **owned** | |
| Lieuwen — *Unsteady Combustor Physics* | **owned** | |
| Natanzon / Culick — *Combustion Instability* | **owned** | Matches `documents/CombustionInstability.md`. |
| Yang et al. — *Liquid Rocket Thrust Chambers* | **owned** | |
| Kuo & Acharya | **owned** | |

NR does not belong on the propulsion physics list. It belongs on the *stiff ODE + conservative PDE* list that those books assume you already know.

---

## 5. Papers (deep research, 2026-08-24)

Grouped by the NR hole they fill. Prefer original papers over blog summaries.

### Integrators — what NR §17 left on the table

1. **Dormand, J. R. & Prince, P. J.** (1980). “A family of embedded Runge-Kutta formulae.” *J. Comput. Appl. Math.* **6** (1) 19–26.  
   The actual DOPRI5 pair. NR §17.2 implements it; Hairer I explains it; this is the source.

2. **Prince, P. J. & Dormand, J. R.** (1981). “High order embedded Runge-Kutta formulae.” *J. Comput. Appl. Math.* **7** 67–75.  
   DOP853. Your `DOPR853Coefficients`.

3. **Gustafsson, K., Lundh, M. & Söderlind, G.** (1988). “A PI stepsize control for the numerical solution of ODEs.” *BIT* **28** 270–287.  
   Why `ComputePIStepSize` exists. NR's `Controller` is a stripped PI (often `beta=0`, i.e. I-only). Hairer I §II.4 plus this paper.

4. **Berry, M. M. & Healy, L. M.** (2004). “Implementation of Gauss-Jackson integration for orbit propagation.” *J. Astronautical Sciences* **52** (3) 331–357.  
   PDF: https://drum.lib.umd.edu/items/e989dac8-3ddb-4377-a549-b041ef77f39b  
   **Read after NR 17.6.** 8th-order Gauss-Jackson is what USSPACECOM special perturbations used for decades. Fixed-step, second-sum, startup via f-and-g series. NR has no implementation.

5. **Jones, B. A. & Anderson, R. L.** (2012). “A survey of symplectic and collocation integration methods for orbit propagation.” AAS 12-214. NTRS 20130000293.  
   When DOPRI5 is the wrong default: long-term, Hamiltonian, collocation / Gauss-Legendre IRK.

6. **Atallah, A. M. et al.** (2019). “Accuracy and efficiency comparison of six numerical integrators for orbit propagation.”  
   Practical bake-off (RK, ABM, Gauss-Jackson, etc.). Use when someone asks “why not RK4.”

7. **Montenbruck, O.** (various) and the force-model chapters in Montenbruck & Gill. If you acquire only one orbit-numerics book, that is it.

### Attitude / estimation — NR is silent

8. **Shuster, M. D. & Oh, S. D.** (1981). “Three-axis attitude determination from vector observations.” *JGCD* **4** (1) 70–77. QUEST.

9. **Lefferts, E. J., Markley, F. L. & Shuster, M. D.** (1982). “Kalman filtering for spacecraft attitude estimation.” *JGCD* **5** (5) 417–429. doi:10.2514/3.56190  
   **The** MEKF paper. Quaternion + gyro bias. NR Ch. 15 will not get you here.

10. **Crassidis, J. L., Markley, F. L. & Cheng, Y.** (2007). “Survey of nonlinear attitude estimation methods.” *JGCD* **30** (1) 12–28.  
    EKF vs UKF vs particle vs nonlinear observers.

11. **Markley, F. L.** (2003). “Attitude error representations for Kalman filtering.” *JGCD* **26** (2) 311–317.  
    Why we use a 3-component error state, not a 4-vector quaternion in the covariance.

### Relative motion / RPO

12. **Clohessy, W. H. & Wiltshire, R. S.** (1960). “Terminal guidance system for satellite rendezvous.” *J. Aerospace Sciences* **27** (9) 653–658.  
    You already used the 2:1 ellipse numbers in True Anomaly prep.

13. **Yamanaka, K. & Ankersen, F.** (2002). “New CW-based relative motion state transition matrix.” *JGCD* **25** (2) 60–66.  
    Eccentric chief. Next after CW.

14. **Fehse, W.** — book, not a paper, listed above. Operational RPO.

### Simulation architecture (this is the C++ job, not NR)

15. **Kenneally, P. W., Piggott, S. & Schaub, H.** (2020). “Basilisk: A flexible, scalable and modular astrodynamics simulation framework.” *J. Aerospace Information Systems* **17** (9) 496–507. doi:10.2514/1.I010762  
    Author PDF: https://hanspeterschaub.info/PapersPrivate/Kenneally2020a.pdf  
    Modules, tasks, message passing, C++ core / Python wrap, FTRT Monte Carlo, 365× realtime goal. **This is the architecture paper to read after NR 17.0.**

16. **Allard, C., Diaz-Ramos, M., Kenneally, P., Schaub, H. & Piggott, S.** (2018). “Modular software architecture for fully coupled spacecraft simulations.” *JAIS* **15** (12). doi:10.2514/1.I010653  
    How to split *coupled* EOMs (flex, slosh, wheels) without assembling one giant RHS by hand — the thing NR `derivs(x,y,dydx)` does not scale to.

17. **Vaz Carneiro, J. & Schaub, H.** (2024). “Scalable architecture for rapid setup and execution of multi-satellite simulations.” *Advances in Space Research*.  
    Multi-vehicle messaging. https://hanspeterschaub.info/PapersPrivate/VazCarneiro2024a.pdf

18. NASA **42**, **Trick**, **GMAT** (Hughes et al. 2014 V&V paper, AIAA 2014-4151), Orekit — know they exist. Basilisk is the one to read as source.

### Stiff chemistry / propulsion numerics

19. **Hindmarsh, A. C. et al.** SUNDIALS / CVODE. Current docs: https://sundials.readthedocs.io/  
    Variable-order BDF. This, not `StepperRoss`, is what Pele/combustion codes call.

20. **Curtis, N. J., Niemeyer, K. E. & Sung, C.-J.** (2017). “An investigation of GPU-based stiff chemical kinetics integration methods.” *Combustion and Flame*. pyJac + implicit RK vs CVODE.

21. **Niemeyer, K. E. et al.** pyJac analytic Jacobians. If we ever revisit the failed PINN kinetics work, the right numerical baseline is CVODE+analytic J, not another neural surrogate.

22. **Stone, C. P. & Davis, R. L.** (2013). “Techniques for solving stiff chemical kinetics on GPUs.” AIAA 2013-0369. CVODE on CUDA.

### Trajectory optimization (when NR §10 runs out)

23. **Betts, J. T.** *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming* (SIAM). Direct transcription.
24. **Rao, A. V. et al.** GPOPS-II / Radau collocation.
25. **Lawden, D. F.** primer vector; **Edelbaum**; **Sims-Flanagan** for low-thrust. Conway (ed.) *Spacecraft Trajectory Optimization*.

---

## 6. Suggested 4-week reading plan

Assumes ~6 focused hours/week, overlapping a sim-software job rather than a sabbatical.

**Week 1 — stepper + events**  
NR 17.0–17.2, Hairer I II.4–II.6, NR 9.1–9.3.  
*Exercise:* event-localized eclipse on the existing Cosmos two-body propagator (dense output + Brent), not a new integrator.

**Week 2 — second order + stiffness + linear algebra**  
NR 17.4–17.5, StormerRule.tex reread, NR 2.3/2.6/2.7. Skim Berry-Healy.  
*Exercise:* component-wise tolerances on a 7-vector `[r,v,m]` and a 7-vector `[q,ω]` — prove a single `eps` is wrong.

**Week 3 — spacecraft physics, not NR**  
Sidi ch. attitude kinematics + gravity gradient; Curtis relative motion; Clohessy-Wiltshire derivation from memory. Start Markley/Crassidis ch. 5 if acquired.  
*Exercise:* Hamilton scalar-first body-to-inertial, write `q̇ = ½ q ⊗ ω` and a gravity-gradient torque, unit-test quaternion norm drift vs RK4 vs DOPRI5.

**Week 4 — architecture**  
Kenneally 2020 + Allard 2018. Sketch how `RKMethods` becomes a *module* with a message interface rather than a standalone `integrate()`. Read NR 3 (tables) and 7 (MC seeds) only as far as that sketch needs.

Propulsion-track fork, if that is the week’s work instead of GNC: swap week 3–4 for Sutton nozzle + NR 17.5 + SUNDIALS docs + Poinsot ch. on operator splitting. Do not read NR 20 as if it were a modern CFD book.

---

## 7. Rewrite order (code, later)

Not this session. When we rewrite, this is the DAG:

1. **Roots** — `Numerical::Roots::{Bisect,Brent,Ridder}` on a callable, with bracket validation, iteration cap, NaN policy. Event locator as a client of dense output, not a sibling of `rtbis`.
2. **Finish RKMethods** — event callbacks, component-wise `atol/rtol` vectors, quaternion-aware error norms (error in the tangent space, not in 4D). Kill or quarantine `ODE/StepperDopr5.h`.
3. **Rosenbrock / IMEX** — `StepperRoss` principles with *our* LU (Stunticons or a dense `Algebra::Solvers::LU`), not `nr3` matrices. Hairer II as the spec.
4. **Interpolation** — you already have `Numerical/Interpolation/`. Promote it; do not import `interp_1d.h`.
5. **Linear algebra** — sparse path exists. Dense LU/QR/SVD: wrap a real library (Eigen / LAPACK) behind `Algebra::Solvers`, do not transcribe `ludcmp.h`.
6. **Störmer / Gauss-Jackson** — second-order orbit path. Störmer is started; Gauss-Jackson is a new module with Berry-Healy as the spec, *after* we admit NR will not help.
7. **Sim orchestrator** — clock, multi-rate, events, logging. Basilisk papers, not NR `Odeint`.

---

## 8. Local file index

| What | Where |
|------|--------|
| NR3 PDF + code | `…/books/Physics/Press W.H., … 2007/{Numerical.Recipes.3ed.pdf,NR_C301/}` |
| Parse products | `…/2007/parsed/` (`MANIFEST.md`, `catalog.json`, `toc.md`, `recipes.md`, full `.txt`) |
| This guide (git) | `Propulsion/documents/NumericalRecipes/READING_GUIDE.md` |
| NR-shaped port (do not extend) | `Cosmos/Source/Numerical/ODE/{StepperDopr5,StepperBase,ODEInt}.*` |
| First-principles RK | `Cosmos/Source/Numerical/ODE/RKMethods/` |
| Hairer audit | `T1000/Cosmos/AUDIT.md` |
| Störmer derivation | `documents/StormerRule.tex` |
| Interview NR subset | `INTERVIEW_PREP_NUMERICAL.md` and Galvatron `TrueAnomaly/FINAL_BATTLECARD_2026-08-20.md` (roots, RK, stiffness, never-invert) |

---

*Started 2026-08-24. Update this file when a chapter is actually worked, a book is acquired, or a rewrite lands.*
