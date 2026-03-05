# Interview Prep: Numerical ODE Methods & Astrodynamics

Covers Numerical Recipes (NR3) design patterns, Runge-Kutta theory, adaptive
stepping, and two-body orbital mechanics. Written for aerospace/SpaceX-track
interviews where C++ numerical simulation experience is expected.

---

## Part 1: Runge-Kutta Methods from Scratch

### What is an ODE integrator doing?

Given **ẏ = f(t, y)** with initial condition **y(t₀) = y₀**, find y(t) for
t > t₀.  Analytically impossible in general → march forward numerically,
step by step.

All RK methods approximate the derivative over a step [tₙ, tₙ+h] by a
**weighted sum of function evaluations** (called stages or k-values):

```
y_{n+1} = y_n + h · Σ bᵢ kᵢ
kᵢ = f(t_n + cᵢh,  y_n + h · Σⱼ aᵢⱼ kⱼ)
```

The coefficients {cᵢ, aᵢⱼ, bᵢ} are the **Butcher tableau**.  Different
choices give different methods.

---

## Part 2: The Stepper Concept (Numerical Recipes §17.2)

### Why "Stepper" instead of just a function?

NR3 encapsulates one trial step as a **struct** (object) because it needs to:
- Hold integration state between steps (x, y, dydx)
- Cache stage k-values (avoid recomputation with FSAL)
- Track step-size history (for PI control)
- Provide error estimate without exposing Butcher details to caller

```cpp
struct StepperBase<N> {
    double x, xold;       // current and previous independent variable
    State  y, dydx;       // state and derivative at x
    State  yout, yerr;    // output and error estimate after do_step()
    double hdid, hnext;   // bookkeeping for adaptive driver
    double atol, rtol;    // tolerances
    double error_norm();  // scaled RMS error (≤1 = accept)
};
```

Two operations:
- **`do_step(h, derivs)`** — one fixed step, fills yout and yerr, no decision
- **`step(htry, derivs)`** — adaptive: calls do_step, rejects if err>1, sets
  hdid and hnext

---

## Part 3: Embedded Pairs and Error Estimation

### The key idea: two answers for the price of ~1

An **embedded** method computes two approximations of different orders in a
single set of k-evaluations:
- y₅ (5th order) — the advance
- y₄ (4th order, "embedded") — not used to advance, only for error

```
yerr = y₅ - y₄  ← local error estimate, costs NO extra f() calls
```

This is why embedded methods dominate in practice over non-embedded (e.g.,
RK4 with no error estimate).

### Why different orders?

- 4th-order error scales as h⁵ → error ∝ h⁵
- 5th-order answer is more accurate → use it to advance
- Use their *difference* as an indicator of the 4th-order error
- This is called **local extrapolation** or **stepping on the higher-order result**

---

## Part 4: Cash-Karp 4(5) — The NR3 Workhorse

### Why Cash-Karp?

Designed by Cash and Karp (1990) to minimize error in the 5th-order solution
(not just the 4th), making local extrapolation especially accurate.

### Butcher Tableau (s=6 stages)

```
c₁=0       │
c₂=1/5     │  a₂₁=1/5
c₃=3/10    │  a₃₁=3/40       a₃₂=9/40
c₄=3/5     │  a₄₁=3/10       a₄₂=−9/10     a₄₃=6/5
c₅=1       │  a₅₁=−11/54     a₅₂=5/2       a₅₃=−70/27    a₅₄=35/27
c₆=7/8     │  a₆₁=1631/55296 a₆₂=175/512   a₆₃=575/13824
           │  a₆₄=44275/110592  a₆₅=253/4096
───────────┼──────────────────────────────────────────────────────────
b  (5th)   │  37/378   0   250/621   125/594   0     512/1771
b* (4th)   │  2825/27648 0  18575/48384  13525/55296  277/14336  1/4
```

Error coefficients `e = b - b*` are used directly: `yerr[i] = h · Σ eⱼ · k[j][i]`.

### Step-size control (NR §17.2.1 — "rkqs")

After computing `err = error_norm()`:

```
If err > 1 (rejected):
    h_new = h · max(0.9 · err^(−1/4), 0.1·h)   // shrink, floor 10×
    retry

If err ≤ 1 (accepted):
    h_next = h · min(0.9 · err^(−1/5), 5·h)    // grow, cap 5×
```

Exponents: −1/(p+1) for shrink (p=4), −1/p for grow (p=5) — derived by
setting the leading error term equal to the tolerance bound.

---

## Part 5: Dormand-Prince 5(4) — DOPRI5

### Two advantages over Cash-Karp

**1. FSAL (First Same As Last)**

k₇ of step n = k₁ of step n+1 (because the 5th-order b-coefficients equal
the 7th row of the a-matrix).  This saves 1 f-evaluation per step (6 instead
of 7 effective evaluations for a 7-stage method).

**2. Dense output**

A degree-4 continuous polynomial can be constructed between tₙ and tₙ₊₁ using
only k-values already computed (plus sometimes one extra midpoint evaluation).
This allows:
- Output at arbitrary times (not just step endpoints)
- Event detection: find t where some condition g(y(t))=0 via bisection
- Non-uniform output without adding steps

### DOPRI5 Butcher Tableau (s=7)

```
c₁=0      │
c₂=1/5    │  1/5
c₃=3/10   │  3/40       9/40
c₄=4/5    │  44/45      −56/15      32/9
c₅=8/9    │  19372/6561  −25360/2187  64448/6561  −212/729
c₆=1      │  9017/3168   −355/33    46732/5247   49/176   −5103/18656
c₇=1(FSAL)│  35/384      0          500/1113     125/192  −2187/6784  11/84
──────────┼──────────────────────────────────────────────────────────────────
b (5th)   │  35/384  0  500/1113  125/192  −2187/6784  11/84  0
b*(4th)   │  5179/57600 0 7571/16695 393/640 −92097/339200 187/2100 1/40
```

Error = b − b* = `[71/57600, 0, −71/16695, 71/1920, −17253/339200, 22/525, −1/40]`

### PI Step Controller (Hairer & Wanner §II.4)

Pure I-control (just tracking current error) causes oscillation in h when the
error fluctuates.  PI control adds a derivative term using the *previous* error:

```
scale = S · err^(−α) · err_prev^(β)
h_new = h · clamp(scale, 0.2, 10)
```

For order-5 method: **α = 0.7/5 = 0.14, β = 0.4/5 = 0.08**, S = 0.9.

The β term acts like a "brake" — if the last error was also small, it allows
more aggressive growth; if the last step was bad, it's more conservative.

---

## Part 6: The ODE Driver (odeint pattern)

```cpp
while (x < x_end) {
    if (x + h > x_end) h = x_end - x;  // don't overshoot
    stepper.step(h, derivs);
    observer(stepper.x, stepper.y);     // record if desired
    h = stepper.hnext;
    if (abs(h) < h_min) throw underflow;
}
```

### Observer pattern
The observer callback decouples the driver from output logic:
- Record all states (trajectory)
- Print progress
- Detect events (root-find using dense output)
- Stop early (throw from observer)

---

## Part 7: Error Norm Design

### Why mixed tolerances?

```
scale_i = atol + rtol · max(|y_i|, |yout_i|)
```

- Pure atol: fine for small y; wasteful for large y (accepts huge relative err)
- Pure rtol: fails near zero (relative error of 1e-6 of 0 = nothing meaningful)
- Mixed: adapts to the actual magnitude.  atol is the "floor", rtol is "how
  many significant digits do you want"

### Scaled RMS norm (not max norm):

```
err = sqrt( (1/N) · Σ (yerr_i / scale_i)² )
```

Max-norm is more conservative (one bad component kills the step); RMS norm is
"average quality" and tends to give larger steps.  NR3 uses RMS.

---

## Part 8: Stiff vs Non-Stiff ODEs

### What is stiffness?

A system is **stiff** if it has widely different timescales — some components
evolve on timescale τ_fast, others on τ_slow, where τ_slow >> τ_fast.

Example: chemical kinetics with fast bond vibrations (ps) and slow reaction
rates (ms); atmospheric re-entry with fast acoustic modes and slow trajectory.

### Why does stiffness break explicit RK methods?

Stability requires: |h · λ_max| ≤ C where C is the stability constant of the
method (for RK4, C ≈ 2.8).  For a stiff system, λ_max is huge, so h must
be tiny even when the *solution* is changing slowly.

→ Explicit RK becomes astronomically slow due to stability, not accuracy.

### Solutions
- **Implicit methods** (Rosenbrock, SDIRK, Radau): solve a nonlinear system
  per step; stability region covers the entire left half-plane
- **LSODA** (MATLAB ode15s): switches between stiff/non-stiff automatically
- **Exponential integrators**: exact for the linear part

Orbital mechanics is generally **non-stiff** (except near singularities or
with certain perturbations like atmospheric drag with very high ballistic
coefficient).

---

## Part 9: Two-Body Orbital Mechanics

### Equation of Motion (Cowell's method)

```
r̈ = -μ/|r|³ · r       (vector form)
```

State vector y = [x, y, z, vx, vy, vz]:
```
ẏ = [vx, vy, vz,  -μx/r³,  -μy/r³,  -μz/r³]
```

This is a 6D autonomous ODE (t does not appear explicitly for pure two-body).

### Gravitational parameter μ = GM

| Body   | μ (m³/s²)       |
|--------|-----------------|
| Earth  | 3.986004418e14  |
| Moon   | 4.9048695e12    |
| Sun    | 1.32712440018e20|

### Conserved quantities (verify integrator accuracy)

```
Energy:     ε  = v²/2 - μ/r           (J/kg, constant)
Angular h:  h  = r × v                 (m²/s vector, constant)
Eccentricity: e_vec = v×h/μ - r̂       (dimensionless vector, constant)
```

Drift in ε or |h| reveals integrator error accumulation.

---

## Part 10: Keplerian Elements

### The 6 elements

| Symbol | Name | Physical meaning |
|--------|------|-----------------|
| a | Semi-major axis [m] | Size of orbit; determines period and energy |
| e | Eccentricity [-] | Shape: 0=circle, <1=ellipse, 1=parabola, >1=hyperbola |
| i | Inclination [rad] | Tilt of orbital plane relative to equatorial |
| Ω | RAAN [rad] | Rotation of orbital plane around Z (vernal equinox) |
| ω | Arg. of periapsis [rad] | Rotation of major axis within orbital plane |
| ν | True anomaly [rad] | Where satellite IS in orbit right now |

### Key relationships

```
p = a(1-e²)                          (semi-latus rectum)
r = p/(1 + e·cos ν)                 (orbit equation)
T = 2π√(a³/μ)                       (Kepler's 3rd law)
ε = -μ/(2a)                         (energy from SMA)
v_circ = √(μ/r)                     (circular orbit speed)
v = √(μ(2/r - 1/a))                 (vis-viva equation)
```

### Mean / Eccentric / True anomaly

```
M = n(t - t_peri)           n = 2π/T = √(μ/a³) (mean motion)
M = E - e·sin(E)            (Kepler's equation, transcendental → Newton's method)
tan(ν/2) = √((1+e)/(1-e)) · tan(E/2)
```

**Newton-Raphson for Kepler's equation:**
```
E₀ = M  (for e < 0.8)  or  E₀ = π (for e ≥ 0.8)
Eₙ₊₁ = Eₙ - (Eₙ - e·sin(Eₙ) - M) / (1 - e·cos(Eₙ))
```
Converges in 3-5 iterations for e < 0.9.

### Singular cases

| Condition | Problem | Fix |
|-----------|---------|-----|
| e ≈ 0 (circular) | ω undefined | Use argument of latitude u = ω + ν |
| i ≈ 0 (equatorial) | Ω undefined | Use longitude of periapsis ϖ = Ω + ω |
| e≈0 and i≈0 | both undefined | Use true longitude L = Ω + ω + ν |
| e ≥ 1 | hyperbolic/parabolic | Different formulas; a < 0 for hyperbola |

---

## Part 11: State Vector ↔ Elements Conversion

### Elements → Cartesian (common in practice: set up initial conditions)

1. Compute r = p/(1 + e·cos ν), speed √(μ/p)
2. Position and velocity in **perifocal (PQW)** frame:
   - P̂ points to periapsis, Q̂ = 90° ahead in orbit plane
   - r_pqw = r·[cos ν, sin ν, 0]
   - v_pqw = √(μ/p)·[−sin ν, e+cos ν, 0]
3. Rotate PQW → ECI: R3(−Ω)·R1(−i)·R3(−ω)

### Cartesian → Elements (needed for analysis/output)

```
h_vec = r × v          → |h|, i = acos(h_z/|h|)
N_vec = ẑ × h_vec      → Ω = acos(N_x/|N|), quadrant from N_y
e_vec = v×h/μ - r̂      → e = |e_vec|, ω from N·e_vec, quadrant from e_z
ε = v²/2 - μ/r         → a = -μ/(2ε)
ν = acos(e_vec·r/(e·r)), quadrant from ṙ·r (if >0: approaching apoapsis)
```

---

## Part 12: Orbit Types and Mission Profiles

| Orbit | Altitude | Period | Characteristics |
|-------|----------|--------|-----------------|
| LEO (ISS) | 400 km | ~92.7 min | Low latency, high drag |
| SSO | 500-800 km | ~96-101 min | Sun-synchronous (ground repeat) |
| MEO (GPS) | 20,200 km | 12 hr | Navigation, less drag |
| GEO | 35,786 km | 24 hr | Fixed over equator |
| GTO | 200×35786 km | ~10.5 hr | Transfer orbit to GEO |
| Molniya | 500×40000 km, i=63.4° | 12 hr | High-latitude coverage |

### Hohmann Transfer (minimum ΔV between two circular orbits)

```
Δv₁ = v_circ(r₁) · (√(2r₂/(r₁+r₂)) - 1)   (burn at r₁)
Δv₂ = v_circ(r₂) · (1 - √(2r₁/(r₁+r₂)))   (circularise at r₂)
Total ΔV = Δv₁ + Δv₂
```

### Why J2 matters (interview favourite)

Earth is oblate (equatorial bulge). The dominant gravitational perturbation
after the central term is J2:

```
a_J2 = -(3/2) · J2 · μ · R_E² / r⁴ · [correction terms in x,y,z]
```

J2 = 1.08263e-3 (dimensionless).  Effects:
- **RAAN drift** (nodal precession): Ω̇ = -(3/2)·n·J2·(R_E/a)²·cos(i)/(1-e²)²
  → choose i=63.4° (Molniya) for zero drift; i≈97.7° for SSO sun-sync
- **Argument of periapsis drift**: ω̇ = 0 at i=63.4° (critical inclination)

---

## Part 13: Numerical Integration for Orbital Mechanics — Practical Notes

### Why DOPRI5 over RK4 for orbits?

1. Adaptive step: tiny h at periapsis, large h at apoapsis automatically.
   RK4 with fixed h either wastes work near apoapsis or loses accuracy near
   periapsis.
2. Error control: DOPRI5 guarantees `atol + rtol·|y|` per step; RK4 has no
   such guarantee.
3. FSAL: 6 effective function evaluations per step (7-stage but first=last).

### Cowell vs Encke

| Method | Approach | When to use |
|--------|----------|------------|
| **Cowell** | Integrate r̈ = f(r,v,t) directly | Universal; simple to implement |
| **Encke** | Integrate Δr = r - r_ref (deviation from reference conic) | Long arcs with weak perturbations; reduces cancellation error |
| **VOP** (Variation of Parameters) | Integrate osculating elements | When perturbations are small; elements change slowly |

### How to verify your integrator is working

1. **Energy conservation**: `|ε(t) - ε(0)| / |ε(0)| < tolerance`
2. **Angular momentum conservation**: `||h(t)| - |h(0)|| / |h(0)| < tolerance`
3. **Period test**: propagate exactly one Keplerian period; check r, |v| match
4. **Grid convergence**: halve h, verify error drops by factor ≈ 2^p (p=order)

### Symplectic integrators

Standard RK methods are **NOT symplectic** — they don't preserve the
symplectic structure of Hamiltonian systems. Over long integrations:
- Energy drifts (even if well-bounded per step)
- Better choice for long-term: **Störmer-Verlet**, **leapfrog**, or
  **symplectic RK** — these conserve a *perturbed* Hamiltonian exactly,
  giving bounded energy error over arbitrary time

For mission-critical long-arc integration (e.g., 100-year asteroid orbit):
use symplectic; for short-arc guidance (1 orbit): DOPRI5 is fine.

---

## Part 14: Common Interview Questions & Answers

**Q: What is a stiff ODE and why do explicit methods fail?**
A: A stiff system has components evolving on very different timescales (λ_max
>> λ_min). Explicit methods have finite stability regions — they require h <
C/|λ_max| for stability, even when the *solution* is varying on the slow
timescale. Use implicit methods (Rosenbrock, BDF) for stiff problems.

**Q: What's the difference between order and stage count?**
A: Order = accuracy (error ∝ h^(p+1)); stage count s = number of f() calls
per step. For explicit RK, order ≤ s, but the Butcher barriers mean for
p≥5, s > p. DOPRI5: s=7 stages, p=5. This is the Butcher barrier.

**Q: What is FSAL and why does it matter?**
A: First Same As Last. The final k₇ equals the first k₁ of the next step (the
b-coefficients equal the last row of a). Saves 1 f-evaluation per accepted
step — DOPRI5 effectively needs only 6 new evaluations per step vs 7 nominal.
Critical for performance when f() is expensive (e.g., full-precision gravity
model).

**Q: How do you choose atol and rtol?**
A: Depends on application. For orbital mechanics with position in metres:
- atol ≈ 1e-3 to 1e-6 (mm to μm, absolute floor)
- rtol ≈ 1e-6 to 1e-9 (relative digits desired)
Rule of thumb: relative error in conserved quantities ≈ rtol.

**Q: What is dense output and when do you need it?**
A: An interpolant polynomial (typically degree 4 for DOPRI5) that gives y at
any t within [tₙ, tₙ₊₁] without additional f() calls. Needed for:
- Event detection (rootfinding in y(t))
- Output at specified times not aligned with adaptive step endpoints
- Continuity in animations/trajectories

**Q: Explain Kepler's equation and how you solve it.**
A: M = E - e·sin(E), M=mean anomaly (linear in time), E=eccentric anomaly
(geometric), ν=true anomaly (where satellite is). Given M (from time), solve
for E via Newton-Raphson (converges in 3-5 iterations for e<0.9). Then
compute ν = 2·atan2(√(1+e)·sin(E/2), √(1-e)·cos(E/2)).

**Q: Why does the ISS orbit decay?**
A: Atmospheric drag at 400 km altitude (ρ ≈ 10⁻¹¹ kg/m³) — small but non-
zero. Drag dissipates energy, lowering the orbit. Semi-major axis decreases,
which *increases* orbital speed (vis-viva), but the total energy decreases.
ISS loses ~2 km/month and needs periodic reboosts.

**Q: What is the J2 perturbation?**
A: Earth's equatorial bulge causes a non-central gravity term proportional to
J2=1.08e-3. Primary effects: precession of RAAN (nodal regression) and drift
of argument of periapsis. RAAN drifts at ~−7°/day for ISS (retrograde). SSO
uses J2 precession deliberately to keep the orbital plane synchronized with
the Sun (Ω̇ ≈ +0.9856°/day eastward to match Earth's orbit).

---

## Part 15: Files in This Repo to Reference

```
T1000/Source/
  Numerical/ODE/
    StepperBase.hpp      ← Abstract base, error norm
    RKCK.hpp             ← Cash-Karp 4(5), NR3 §17.2 workhorse
    DOPRI5.hpp           ← Dormand-Prince 5(4), FSAL + PI control
    ODEDriver.hpp        ← integrate_adaptive with observer
  OrbitalMechanics/
    Constants.hpp        ← μ, R_E, AU, etc.
    StateVector.hpp      ← [x,y,z,vx,vy,vz] + helpers
    TwoBody.hpp          ← Cowell EOM functor + period/speed formulas
    OrbitalElements.hpp  ← elements↔state, Kepler solver
    Propagator.hpp       ← High-level propagate() + conservation tracking

T1000/T1000/numerical/  ← Original Python implementations (reference)
  RungeKuttaMethod.py
  RKMethods/DOPRI5Coefficients.py
  RKMethods/CalculateNewYAndError.py
  RKMethods/ComputePIStepSize.py
  RKMethods/CalculateScaledError.py
```

Run tests in Docker:
```bash
cd InServiceOfX/Deployments/DockerContainers/Builds/Physics/PropulsionWithCUDA
docker_builder run_build_tests.yml
```
