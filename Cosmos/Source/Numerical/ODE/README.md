Shared ODE integration
======================

`RKMethods/` contains the configurable stage calculation, error scaling,
PI step control and integration drivers. `DOPRI5Coefficients` supplies the
Dormand–Prince 5(4) pair. `T1000/Source/OrbitalMechanics/Propagator.hpp` now
uses these components; its former separate `DOPRI5.hpp` has been removed.
The orbit API and physics remain in OrbitalMechanics. The fixed-step
`Astrodynamics/Propagators/NumerovOrbit.h` remains a distinct method.

Link **CosmosODE** to get the DOPRI5 coefficient definitions and Cosmos public
include path. Its standalone CMake directory is `RKMethods/`; it does not
require the examples, GoogleTest or CoolProp. The larger Numerical target
links it publicly. The orbital demo demonstrates consuming it from T1000.

`CalculateNewYAndError` supports fixed arrays, valarray and NVector. The array
path keeps state dimension separate from stage count and reuses the existing
output-parameter calculation. The driver expects an RHS returning a state.
`StepWithPIControl` owns its calculation object; the calculation borrows its
coefficient objects, which must outlive it. Exported DOPRI5 coefficients are
currently double. A templated kernel alone does not imply arbitrary-precision
or CUDA support.

`IntegrateWithPIControl::integrate_with_observer<N>` takes IntegrationInputs
and an observer `(time, const state&, accepted_step)`. It returns a tuple of
final time, final state and accepted-step count. Observers run only after an
accepted step. No trajectory is allocated by this path. `integrate<N>` uses
the same loop and additionally stores the initial state and accepted steps.

Both paths normalize the step direction, clamp trials to the endpoint, reset
PI history for a new integration and handle an empty interval without calling
the RHS. Invalid/nonfinite inputs, unrepresentable progress and exhausted step
budgets raise exceptions; an incomplete integration is not returned as success.
Local error tolerances remain estimates, not global-error guarantees. The
older dense-output and higher-order integration drivers have separate code
paths and should not be assumed to share these endpoint guarantees.

Solver regression tests belong in `UnitTests/Numerical/ODE/RKMethods` under
Cosmos/Source. Orbital physics/trajectory tests remain in
T1000/Source/UnitTests/OrbitalMechanics. Cash–Karp and its supporting T1000
driver/tests remain because that method has not been migrated.
