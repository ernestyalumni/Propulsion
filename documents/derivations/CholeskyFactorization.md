# Cholesky factorization, from first principles

**Serves:** covariance square roots (P = L Lᵀ) for Monte Carlo sampling and
square-root Kalman filtering; normal-equation solves in least squares;
mass-matrix solves in multi-body dynamics.
**NR reference:** §2.9, printed p. 100 (PDF 124). Substitutes for the
stability argument: Higham, *Accuracy and Stability of Numerical Algorithms*,
2nd ed., ch. 10; Trefethen and Bau, *Numerical Linear Algebra*, lecture 23.
**Module:** `Cosmos/Rust/cosmos_numerical/src/linear_algebra/cholesky.rs`.

## 1. The physics that produces the matrix

A covariance matrix P = E[(x − μ)(x − μ)ᵀ] is symmetric by construction and
positive semidefinite because vᵀPv = E[(vᵀ(x − μ))²] ≥ 0 for every v. It is
positive definite when no linear combination of the state has zero variance,
which is the generic case for a navigation state. A mass matrix M(q) of a
multi-body system is symmetric positive definite because kinetic energy
½ q̇ᵀ M q̇ is positive for every nonzero velocity. Normal equations AᵀA x = Aᵀb
have a symmetric positive definite matrix whenever A has full column rank.

So the property we are handed is: **A is symmetric and vᵀAv > 0 for all
v ≠ 0.** Everything below follows from that alone.

## 2. Existence and uniqueness

Claim: a symmetric positive definite A ∈ ℝⁿˣⁿ has exactly one factorization
A = L Lᵀ with L lower triangular and positive diagonal.

Write A in block form with a₁₁ the first pivot, a the first column below it,
and A₂₂ the trailing block:

    A = [ a₁₁   aᵀ ]
        [ a     A₂₂ ]

Positive definiteness with v = e₁ gives a₁₁ > 0, so ℓ₁₁ = √a₁₁ is real. Set
ℓ = a / ℓ₁₁. Then

    A = [ ℓ₁₁  0 ] [ 1   0             ] [ ℓ₁₁  ℓᵀ ]
        [ ℓ    I ] [ 0   A₂₂ − ℓ ℓᵀ    ] [ 0    I  ]

and the Schur complement S = A₂₂ − ℓ ℓᵀ = A₂₂ − a aᵀ / a₁₁ is again symmetric
positive definite: for any w ≠ 0 take v = (−aᵀw / a₁₁, w) and compute
vᵀAv = wᵀSw > 0. Induction on n finishes existence. Uniqueness: if
L₁L₁ᵀ = L₂L₂ᵀ then L₂⁻¹L₁ = L₂ᵀL₁⁻ᵀ is both lower and upper triangular, hence
diagonal, and equal to its own inverse transpose, so its entries are ±1; the
positive-diagonal convention fixes the sign.

## 3. The recurrence

Equate entries of A = L Lᵀ column by column. For column j, using only entries
already determined (columns 1 … j − 1 of L):

    ℓ_jj = sqrt( a_jj − Σ_{k<j} ℓ_jk² )
    ℓ_ij = ( a_ij − Σ_{k<j} ℓ_ik ℓ_jk ) / ℓ_jj        for i > j

The quantity under the square root is the (j, j) entry of the j-th Schur
complement, so by §2 it is strictly positive when A is positive definite.
**If it is ≤ 0 at any j, A is not positive definite, and j is the first index
where that can be shown.** That is the algorithm's built-in test for the
property; it costs nothing extra and it is why Cholesky is the right way to
check whether a covariance is still valid after an update.

Cost: n³/3 flops, half of LU, because symmetry halves the work and no
pivoting is needed.

## 4. Why no pivoting is needed (stability)

For general LU, pivoting bounds element growth. For Cholesky, growth is
bounded by the matrix itself: from the diagonal recurrence,
Σ_{k≤j} ℓ_jk² = a_jj, so every |ℓ_ik| ≤ √a_ii. The computed factor satisfies
L̂ L̂ᵀ = A + ΔA with |ΔA| ≤ c(n) u |L̂||L̂ᵀ| ≤ c(n) u times a matrix whose
entries are bounded by √(a_ii a_jj) (Higham, Theorem 10.3). Backward
stability holds without any pivoting, which is what makes the factorization
cheap enough to run inside a filter every step.

The one caveat that matters for navigation: the *forward* error is governed
by the condition number κ(A) = κ(L)², so a covariance whose eigenvalues span
many decades (position in meters next to a bias in microradians) should be
scaled to comparable units before factoring. That is a units decision, made
in the caller, not inside the factorization.

## 5. Using the factor

- **Solve A x = b:** forward-substitute L y = b, then back-substitute Lᵀ x = y.
  Each is n² flops. The factorization is done once and reused.
- **Determinant and log-determinant:** det A = (Π ℓ_jj)², so
  log det A = 2 Σ log ℓ_jj, computed without overflow. This is the Gaussian
  log-likelihood's normalization term.
- **Sampling from N(μ, P):** with z ~ N(0, I), x = μ + L z has covariance
  L I Lᵀ = P. This is exactly how a Monte Carlo dispersion draws correlated
  initial conditions (NR §7.4 uses this).
- **Mahalanobis distance:** (x − μ)ᵀ P⁻¹ (x − μ) = ‖L⁻¹(x − μ)‖², one forward
  substitution, no inverse.

## 6. What the tests assert

1. Reconstruction: ‖L Lᵀ − A‖_max ≤ 8 n u ‖A‖_max for random SPD A built as
   BᵀB + nI.
2. Solve: A x = b recovers a known x to a relative error consistent with κ(A) u.
3. Non-positive-definite input returns an error naming the first failing
   pivot index and its value; it does not panic and does not return NaN.
4. Log-determinant matches the product of eigenvalues on a diagonal matrix.
5. Sampling: the sample covariance of L z over many draws converges to P at
   the 1/√N rate.

## 7. What NR got right, and what to leave

NR §2.9 states the recurrence and the "fails if not positive definite"
property correctly and recommends Cholesky for exactly the covariance and
normal-equation cases above. Its implementation returns the factor through a
member of a class holding a copy of A and reports failure by throwing a
string. The rewrite returns a factorization type that owns L only, reports
failure as a typed error carrying the pivot index, and never keeps a copy of A.
