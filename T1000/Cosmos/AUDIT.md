# Audit: T1000/Devastator/Source/Numerical — Runge-Kutta Implementation

**Auditor:** Ernest Shackleton (automated review, 2026-03-07)
**Scope:** `T1000/Devastator/Source/Numerical/ODE/RKMethods/`
**Reference:** Hairer, Norsett & Wanner, *Solving ODEs I* (2nd ed., 1993), §II.4–II.6.

---

## Executive Summary

The Devastator library is a **modular, template-heavy** implementation of adaptive
Runge-Kutta methods (DOPRI5, DOP853, RKCK). It separates every concern into its own
class (`CalculateNewYAndError`, `ComputePIStepSize`, `CalculateScaledError`, etc.).
The overall architecture is sound and the primary integration path
(`CalculateNewYAndError`) is numerically **correct**. One latent bug exists in the
superseded `CalculateNextStep` class; all other flagged points are design observations
rather than correctness errors.

---

## Finding 1 — BUG: Off-by-one in `CalculateNextStep::sum_beta_and_k_products()`

**File:** `ODE/RKMethods/RungeKuttaMethod.h`, line 217
**Severity:** **HIGH** (incorrect RK stage evaluations)
**Status:** Effectively superseded by `CalculateNewYAndError`; not used in production
integration path.

### What the code does

`CalculateNextStep` is the older, more-generic RK step calculator.
`sum_beta_and_k_products()` is supposed to compute the inner stage sum:

```
x_l  =  x_n  +  h · ∑_{j=1}^{l-1}  β_{lj} · k_j
```

The summation must run for `j = 1, 2, ..., l-1`.
The `j=1` term is handled *before* the loop (hardcoded via `get_beta_ij(l, 1)`).
The loop is therefore responsible for `j = 2, 3, ..., l-1`.

### The defect

```cpp
// RungeKuttaMethod.h, line 217
for (std::size_t j {2}; j < l - 1; ++j)   // ← BUG: should be j <= l-1 (i.e. j < l)
```

The condition `j < l - 1` stops **one term early**:

| Stage `l` | Terms needed (`j=2..l-1`) | Loop executes (`j=2..l-2`) | Missing |
|-----------|--------------------------|---------------------------|---------|
| 2         | none (j=1 already done)  | none                      | —       |
| 3         | j=2                      | never (2 < 1 is false)    | **j=2** |
| 4         | j=2, j=3                 | j=2 only (2 < 2 is false) | **j=3** |
| 5         | j=2..4                   | j=2,3 only                | **j=4** |

For a 4-stage method (e.g. RK4): stage 4 misses the `β_{43}·k_3` contribution.
For a 7-stage DOPRI5: stages 3–7 all receive incorrect `x_l` values, producing
wrong `k_l` evaluations and therefore a **completely wrong numerical solution**.

### Why production results were not obviously broken

`CalculateNextStep` is **not** called by the adaptive integration stack. The adaptive
path uses `CalculateNewYAndError` exclusively, which has its own (correct) inner loop:

```cpp
// CalculateNewYAndError.h, line 262 — CORRECT
for (std::size_t j {2}; j < l; ++j)   // j = 2 ... l-1 inclusive ✓
```

The two classes coexist in the codebase. `CalculateNextStep` is the older, non-FSAL
fixed-step class that appears to have been used during initial development and unit
testing but was replaced by `CalculateNewYAndError` before production integration.

### Recommendation

If `CalculateNextStep` is retained for any reason, fix line 217:

```cpp
// Before (buggy):
for (std::size_t j {2}; j < l - 1; ++j)

// After (correct):
for (std::size_t j {2}; j < l; ++j)
```

---

## Finding 2 — FSAL Implementation in `CalculateNewYAndError`

**File:** `ODE/RKMethods/CalculateNewYAndError.h`
**Severity:** Observation (correctly implemented)

### What FSAL means

DOPRI5 has the **First-Same-As-Last** property: the 7th stage evaluation
`k_7 = f(t + h, y_{n+1})` equals the `k_1` evaluation of the *next* step (since
`c_7 = 1` and the 5th-order solution `y_{n+1}` is the starting point of the next
step). This saves one function evaluation per accepted step (~14% total cost reduction).

### How Devastator implements it

FSAL is handled **implicitly** via the surrounding `StepWithPIControl` /
`HigherOrderStepWithPIControl` infrastructure:

```
StepInputs.dydx_n_  ←  k_coefficients_.get_ith_coefficient(S)   // = k_7 of step n
```

After an accepted step, the caller copies `k_S` (the last stage value, index `S`)
back into `StepInputs.dydx_n_`. On the next call to `calculate_new_y()`, line 77:

```cpp
k_coefficients.ith_coefficient(1) = initial_dydx;   // = old k_S = new k_1
```

This correctly initialises `k_1` of the new step from the saved final stage.
The loop `for (std::size_t l {2}; l <= S; ++l)` then computes only `k_2 ... k_S`,
yielding `S-1 = 6` new function evaluations per accepted step (not 7).

**Conclusion:** FSAL is correctly exploited. The implementation is subtle because the
responsibility is split across `CalculateNewYAndError` (which sees `initial_dydx`) and
the calling layer (which sets `StepInputs.dydx_n_ = k_S`). This is a valid design
choice — separation of concerns — but requires careful documentation at the callsite.

---

## Finding 3 — `CCoefficients<S>` Size Convention

**File:** `ODE/RKMethods/Coefficients/CCoefficients.h`
**Severity:** Observation (design decision, not a bug)

### What `CCoefficients<S>` stores

```cpp
template <std::size_t S, typename Field = double>
class CCoefficients : public std::vector<Field>
{
  CCoefficients(const std::initializer_list<Field>& c_coefficients):
    std::vector<Field>(S - 1)   // ← allocates S-1 elements
  {
    assert(S - 1 == c_coefficients.size());
    ...
  }

  Field get_ith_element(const std::size_t i) const
  {
    assert(i >= 2 && i <= S);
    return this->operator[](i - 2);   // element 0 → c_2, element 1 → c_3, ...
  }
};
```

`CCoefficients<7>` stores **6 values** (`S-1 = 6`), indexed as `c_2 ... c_7`.

### Is this correct for DOPRI5?

**Yes.** In the standard Butcher tableau, `c_1 = 0` always (the first stage is
evaluated at the current point, no offset needed). Storing `c_1` would be redundant.
By convention the library stores only `c_2 ... c_S` (the S-1 non-trivial node values).

For DOPRI5 (`S=7`), `CCoefficients<7>` is initialised with:

```cpp
{0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0}   // c_2 ... c_7  (6 values)
```

`get_ith_element(i)` maps `i ∈ {2,...,S}` → array index `i-2 ∈ {0,...,S-2}`, which
is correct. `get_ith_element(1)` would assert-fail as expected (c_1 is never needed).

**Summary:** The `S-1` sizing convention is intentional and correct. The initializer
list in `DOPRI5Coefficients.cpp` must supply exactly `S-1` values — verifiable via
the `assert(S-1 == c_coefficients.size())` guard.

---

## Finding 4 — `ComputePIStepSize`: Post-Rejection Cap

**File:** `ODE/RKMethods/ComputePIStepSize.h`
**Severity:** Observation (correct Hairer recommendation)

The PI step-size controller is:

```
h_new = h · S · err^{-α} · err_prev^{β}
```

After a **rejected** step, the code caps the scale factor at 1:

```cpp
return was_rejected
    ? h * std::min(scale, static_cast<Field>(1))   // ← cap scale ≤ 1 after rejection
    : h * scale;
```

This implements **Hairer/Wanner §II.4 recommendation**: after a rejection, never
allow `h` to *increase*, because the PI derivative term `err_prev^β` can momentarily
suggest a larger step even immediately after a rejection (since `err_prev` was from the
last *accepted* step, not the rejected one). Capping at 1 prevents this oscillation.

**Alpha / Beta values** are **caller-supplied** (not hardcoded), which is correct
design. Typical DOPRI5 values:

| Parameter | Symbol | Typical value | Rationale |
|-----------|--------|---------------|-----------|
| `alpha`   | α      | 0.7/5 = 0.14  | Dominant integral term; order p=5 |
| `beta`    | β      | 0.4/5 = 0.08  | Derivative (Lund stabilisation) |
| `safety`  | S      | 0.9           | Prevents step from growing to exactly the stability boundary |
| `min_scale` | —    | 0.2           | Prevents catastrophic shrinkage |
| `max_scale` | —    | 5.0           | Prevents dangerous growth |

The defaults in the constructor (`min=0.2, max=5.0, safety=0.9`) are reasonable, but
callers must supply the correct `alpha` and `beta` for their chosen method order.
Using default α=β=0 would reduce to a pure safety-scaled I-controller — less stable.

---

## Finding 5 — `CalculateScaledError`: Mixed atol+rtol Form

**File:** `ODE/RKMethods/CalculateScaledError.h`
**Severity:** Observation (correctly implements Hairer recommendation)

```cpp
const Field scale {
    a_tolerance_ +
      r_tolerance_ * std::max(std::abs(y_0[i]), std::abs(y_out[i]))};

error += (y_err[i] / scale) * (y_err[i] / scale);
```

Then:

```cpp
return std::sqrt(error / N);   // RMS scaled error norm
```

This matches **Hairer/Wanner §II.4, equation (4.11)**:

```
sc_i = atol + rtol · max(|y_0^i|, |y_{n+1}^i|)
err  = sqrt( (1/N) · ∑ (y_err_i / sc_i)² )
```

Using `max(|y0|, |yout|)` rather than just `|y0|` or `|yout|` is the recommended
form: it avoids catastrophic loss of significance near zero crossings (where either
endpoint alone could be near zero). The RMS norm (divide by N) normalises the error
metric to be dimension-independent (err ≤ 1 ↔ step accepted regardless of system
size N).

**Conclusion:** Correct.

---

## Finding 6 — Relationship Between Devastator and T1000/Source

| Aspect | Devastator (`T1000/Devastator/Source/Numerical/`) | T1000/Source (`T1000/Source/Numerical/ODE/`) |
|--------|--------------------------------------------------|---------------------------------------------|
| Style | Modular, class-per-concern, `.h`+`.cpp` split | Header-only, `struct`-based, single file per concept |
| Generality | Highly generic: any `Field`, any `ContainerT` | Fixed `double`, `std::array<double,N>` |
| FSAL | Implemented via `StepInputs.dydx_n_` plumbing | Explicit: `this->dydx = k7` in DOPRI5 |
| PI controller | External `ComputePIStepSize` class | Inlined in `DOPRI5::step()` |
| Dense output | `CalculateDenseOutputCoefficient.h` | `DOPRI5::dense_output(theta)` |
| Coefficient tables | Separate `.cpp` files per method | Inline `static constexpr` in the stepper |
| Build system | Requires Devastator CMake + Algebra/ dependency | Standalone, zero external deps |
| Purpose | Original research/exploration library | Clean 2025 interview-ready rewrite |

The Devastator library represents the **original, production-hardened** codebase built
for generality and extensibility. The T1000/Source library is a purposeful **rewrite
for clarity**: every design decision is visible in a single file, every formula
references Hairer directly, and the code compiles standalone. Both implement the same
underlying mathematics (DOPRI5 Dormand-Prince 1980 method) correctly.

---

## Summary Table

| # | File | Issue | Severity | Verdict |
|---|------|-------|----------|---------|
| 1 | `RungeKuttaMethod.h:217` | `j < l-1` should be `j < l` in `sum_beta_and_k_products` | HIGH | Bug (superseded class) |
| 2 | `CalculateNewYAndError.h:77` | FSAL via `initial_dydx = k_S` | INFO | Correct, subtle |
| 3 | `CCoefficients.h` | Stores S-1 values (c₂…cₛ), not S | INFO | Correct by design |
| 4 | `ComputePIStepSize.h:64` | Post-rejection cap `scale ≤ 1` | INFO | Correct (Hairer rec.) |
| 5 | `CalculateScaledError.h:40` | Mixed atol+rtol·max(|y0|,|yout|) | INFO | Correct (Hairer §II.4) |

**Overall verdict:** The active integration path is numerically correct. The one true
bug (`sum_beta_and_k_products`) is in a dead code path. The library is safe to use as-is
for production DOPRI5 adaptive integration.
