# Exact Runge–Kutta coefficient calculations

Run from this directory:

```sh
python3 -B verify.py
rustc --edition=2021 exact_rk.rs -o /tmp/propulsion-exact-rk
/tmp/propulsion-exact-rk
```

Requires Python 3 and a Rust compiler; no third-party packages, Cargo workspace
changes, network access or Fortran. Build products go to temporary directories.
The Python verification uses explicit exceptions and also works with `python -O`.

`exact_rk.rs` provides reusable public functions `conditions`, `derive_weights`,
`abscissae`, and `stability`. Import it with `#[path = ".../exact_rk.rs"] mod exact_rk;`
for experiments, or compile it directly as above. Its CLI prints tab-separated
method name, order, c, b, and ascending coefficients of the stability polynomial.

For a supplied strictly lower-triangular stage matrix A, the program computes
c=A*1 and constructs all rooted-tree equations b^T Phi(t)=1/gamma(t).
Exact Gaussian elimination derives a unique b or reports inconsistent or
underdetermined conditions. It does not silently pick free parameters.
This is an analytic calculation over rational numbers, rather than a fit to
floating-point trajectories or merely a lookup of stored output weights.
The CLI examples supply known stage matrices; it does not solve the nonlinear
problem of finding A, or design arbitrary high-order embedded pairs.

Supported examples: Euler, explicit midpoint, Heun order 2, Ralston order 2,
Kutta order 3, classical RK4, and RK4's 3/8 rule. For every example the CLI checks
all conditions through its stated order and detects a violation at the next
order. The independent `verify.py` derives the same weights using the explicitly
written eight conditions through order four, compares known textbook weights,
and compares c, b, and the full stability polynomial with Rust output. Checking
a stability polynomial alone would not establish nonlinear order. A three-stage
counterexample matches exp(z) through degree three but fails b*c^2=1/3;
both implementations explicitly reject it.

The Rust generator supports orders 1 through 8 (200 trees in total through 8).
Its rational arithmetic uses normalized i128 fractions with checked operations.
An intermediate overflow panics even in optimized builds; values never silently
wrap. This is bounded exact arithmetic, not an arbitrary-precision CAS.
Large-denominator high-order tableaux may exceed that bound. `Rational::new`
also panics for a zero denominator or an unrepresentable sign normalization.
Python uses arbitrary-precision `fractions.Fraction`.

The seven Rust tests cover known weights, next-order inconsistency, rooted-tree
counts, malformed matrices, unsupported orders, nonunique systems, fraction
normalization, division and explicit arithmetic failure. The Python check is an
independent low-order oracle, not exhaustive validation of arbitrary tableaux.

## Sources and relationship to existing work

Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*, second
revised edition (local 2008 printing), §II.1 (general formulation and fourth-order
conditions, particularly Theorem 1.6, printed p.137), and §II.2 (rooted-tree order
conditions). Consult the original PDF in `Data/Public/books/EngineeringPhysics/`;
its first Runge–Kutta chapter was directly consulted for this implementation.

Existing C++ coefficient tables are in
`Cosmos/Source/Numerical/ODE/RKMethods/Coefficients/`; Python tables are in
`T1000/T1000/numerical/RKMethods/`. Those files primarily store coefficients.
This new tool derives output weights from the stage matrix. It supplements the
Sage checks in `documents/notes/checks/runge_kutta.sage` and the floating-point
rooted-tree tests in `cosmos_numerical/src/ode/runge_kutta/order_conditions.rs`.
It does not alter the solver's tables or the active Cargo workspace.

The adjacent `../hairer_reference/` harness uses Hairer's original Fortran to
regenerate independent numerical fixtures for the Rust adaptive solvers. That
is a separate test-data generator; these exact calculations do not use it.
