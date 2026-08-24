# Rewriting NR C++ in our architecture

NR3's scientific *content* is useful. NR3's C++ is not a style to emulate. This note is the type-level contract for a first-principles rewrite. The reading order is in `READING_GUIDE.md`.

## Do not extend the NR-shaped port

These files are a transcription of NR §17.2:

- `Cosmos/Source/Numerical/ODE/StepperDopr5.h` (comment cites a GitHub NR dump)
- `StepperBase.h`
- `ODEInt.h` / `Output.h`

New work belongs under `Numerical::ODE::RKMethods` (and sibling namespaces), matching Stunticons `Algebra::`.

NR's license is not a production license. Re-derive.

## What NR §17.0.1 got right

Four layers:

1. **Driver** — start/stop, store, user interface
2. **Stepper** — take the largest \(h\) compatible with the error test
3. **Output** — including dense output at user times
4. **Algorithm** — dumb: \(y_n, h \mapsto y_{n+1}, y_{\mathrm{err}}\)

NR then collapsed 2–4 into one `struct StepperDopr5 : StepperBase` with reference members into the driver. We keep the layers as **separate types**.

## Type map

| NR | Ours |
|----|------|
| `Doub` / `Int` / `VecDoub` | `Field = double`, `std::size_t`, `NVector<N,Field>` or `ContainerT` |
| `using namespace std` | never |
| `#define throw(message)` | real exceptions or `std::optional` / error codes on hot paths; no macros |
| `NRvector` raw `new[]` | `std::vector` / `NVector`; Rule of Zero |
| `template<class D> struct StepperDopr5` | `template<size_t S, typename DerivativeType, typename Field>` |
| Butcher numbers in `dy()` | `ACoefficients<S,Field>` etc. |
| `Controller::success` | `ComputePIStepSize` + `CalculateScaledError` |
| `Odeint::integrate` | `IntegrateWithPIControl` |
| `dense_out` | `CalculateDenseOutputCoefficient` |
| `rtbis` / `zbrent` | `Numerical::Roots::*` (not written yet) |
| `LUdcmp` | wrap Eigen/LAPACK behind `Algebra::Solvers`; Stunticons already has BiCGSTAB on CSR |
| `derivs(x,y,dydx)` inheritance convention | `DerivativeType&&` forwarded callable; no `std::function` on the RHS hot path |

## Invariants we will not inherit

- **Reference members** to caller-owned `x,y,dydx`. Use `StepInputs` values.
- **Virtual dispatch on the RHS** or on the stepper in the inner loop.
- **One global `eps`.** Component-wise `atol_i + rtol_i * scale_i`. Quaternion error in the tangent space.
- **Implicit matrix inverse.** Factor and `solve`.
- **Events as “the user can just stop.”** Sign change → dense-output root → cut step → discrete jump → restart.
- **`using namespace std` and C headers** (`stdlib.h`, `string.h`).

## First concrete rewrite (when we start coding)

`Numerical::Roots::Brent` + event locator on existing DOPRI5 dense output, GoogleTest against a known zero of \(g(t)=\sin(t)\) on an SHO. That closes NR §9 + §17.2 against a spacecraft-sim need without touching 160 other headers.
