//! Cholesky factorization `A = L Lᵀ` of a symmetric positive definite matrix.
//!
//! Derivation, stability argument, and the physics that produces such matrices
//! (covariances, mass matrices, normal equations):
//! `documents/derivations/CholeskyFactorization.md`. NR §2.9, printed p. 100.
//!
//! Only the lower triangle of the input is read, as LAPACK's `dpotrf('L')`
//! does; symmetry is the caller's promise. A non-positive pivot is reported
//! with its index, which is the first place positive definiteness can be
//! shown to fail.

use crate::field::RealField;

/// The input is not positive definite: the Schur-complement pivot at
/// `pivot_index` was not strictly positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotPositiveDefinite<T>
{
  pub pivot_index: usize,
  pub pivot_value: T,
}

/// The lower-triangular factor `L`, stored dense with zeros above the
/// diagonal, so that `A = L Lᵀ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CholeskyFactorization<T: RealField, const N: usize>
{
  lower: [[T; N]; N],
}

impl<T: RealField, const N: usize> CholeskyFactorization<T, N>
{
  /// Column-by-column recurrence from the derivation note §3:
  /// `l_jj = sqrt(a_jj − Σ_{k<j} l_jk²)`,
  /// `l_ij = (a_ij − Σ_{k<j} l_ik l_jk) / l_jj` for `i > j`.
  pub fn factorize(a: &[[T; N]; N]) -> Result<Self, NotPositiveDefinite<T>>
  {
    let mut lower = [[T::zero(); N]; N];

    for j in 0..N
    {
      let mut diagonal = a[j][j];
      for k in 0..j
      {
        diagonal = diagonal - lower[j][k] * lower[j][k];
      }
      if !(diagonal > T::zero()) || !diagonal.is_finite()
      {
        return Err(NotPositiveDefinite { pivot_index: j, pivot_value: diagonal });
      }
      let pivot = diagonal.square_root();
      lower[j][j] = pivot;

      for i in (j + 1)..N
      {
        let mut off_diagonal = a[i][j];
        for k in 0..j
        {
          off_diagonal = off_diagonal - lower[i][k] * lower[j][k];
        }
        lower[i][j] = off_diagonal / pivot;
      }
    }

    Ok(Self { lower })
  }

  pub fn lower(&self) -> &[[T; N]; N]
  {
    &self.lower
  }

  /// Solve `L y = b` by forward substitution. This is also `L⁻¹ b`, the map
  /// that whitens a correlated vector: the Mahalanobis norm of `b` is `‖y‖`.
  pub fn solve_lower(&self, b: &[T; N]) -> [T; N]
  {
    let mut y = [T::zero(); N];
    for i in 0..N
    {
      let mut sum = b[i];
      for k in 0..i
      {
        sum = sum - self.lower[i][k] * y[k];
      }
      y[i] = sum / self.lower[i][i];
    }
    y
  }

  /// Solve `Lᵀ x = y` by back substitution.
  pub fn solve_upper(&self, y: &[T; N]) -> [T; N]
  {
    let mut x = [T::zero(); N];
    for i in (0..N).rev()
    {
      let mut sum = y[i];
      for k in (i + 1)..N
      {
        sum = sum - self.lower[k][i] * x[k];
      }
      x[i] = sum / self.lower[i][i];
    }
    x
  }

  /// Solve `A x = b`.
  pub fn solve(&self, b: &[T; N]) -> [T; N]
  {
    let y = self.solve_lower(b);
    self.solve_upper(&y)
  }

  /// `L z`. With `z ~ N(0, I)` the result has covariance `A`, which is how a
  /// Monte Carlo dispersion draws correlated initial conditions.
  pub fn transform(&self, z: &[T; N]) -> [T; N]
  {
    let mut x = [T::zero(); N];
    for i in 0..N
    {
      let mut sum = T::zero();
      for k in 0..=i
      {
        sum = sum + self.lower[i][k] * z[k];
      }
      x[i] = sum;
    }
    x
  }

  /// `log det A = 2 Σ log l_jj`, computed without forming the determinant.
  pub fn log_determinant(&self) -> T
  {
    let mut sum = T::zero();
    for j in 0..N
    {
      sum = sum + self.lower[j][j].natural_logarithm();
    }
    sum + sum
  }

  /// `L Lᵀ`, for tests and diagnostics.
  pub fn reconstruct(&self) -> [[T; N]; N]
  {
    let mut a = [[T::zero(); N]; N];
    for i in 0..N
    {
      for j in 0..N
      {
        let mut sum = T::zero();
        for k in 0..N
        {
          sum = sum + self.lower[i][k] * self.lower[j][k];
        }
        a[i][j] = sum;
      }
    }
    a
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  /// A small deterministic generator so the tests need no crate and are
  /// reproducible bit for bit (Knuth's MMIX linear congruential constants).
  struct Lcg(u64);

  impl Lcg
  {
    fn next_u64(&mut self) -> u64
    {
      self.0 = self.0
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      self.0
    }

    fn uniform(&mut self) -> f64
    {
      (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn standard_normal(&mut self) -> f64
    {
      let u1 = self.uniform().max(1.0e-300);
      let u2 = self.uniform();
      (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
  }

  /// `BᵀB + n I` is symmetric positive definite for any `B`.
  fn random_spd<const N: usize>(lcg: &mut Lcg) -> [[f64; N]; N]
  {
    let mut b = [[0.0; N]; N];
    for row in b.iter_mut()
    {
      for value in row.iter_mut()
      {
        *value = 2.0 * lcg.uniform() - 1.0;
      }
    }
    let mut a = [[0.0; N]; N];
    for i in 0..N
    {
      for j in 0..N
      {
        let mut sum = 0.0;
        for k in 0..N
        {
          sum += b[k][i] * b[k][j];
        }
        a[i][j] = sum + if i == j { N as f64 } else { 0.0 };
      }
    }
    a
  }

  fn max_abs_entry<const N: usize>(a: &[[f64; N]; N]) -> f64
  {
    a.iter().flatten().fold(0.0, |m, v| m.max(v.abs()))
  }

  #[test]
  fn reconstructs_the_input_to_machine_precision()
  {
    let mut lcg = Lcg(20260901);
    for _ in 0..50
    {
      let a = random_spd::<6>(&mut lcg);
      let factorization = CholeskyFactorization::factorize(&a).unwrap();
      let r = factorization.reconstruct();
      let mut difference = 0.0f64;
      for i in 0..6
      {
        for j in 0..6
        {
          difference = difference.max((r[i][j] - a[i][j]).abs());
        }
      }
      let bound = 8.0 * 6.0 * f64::EPSILON * max_abs_entry(&a);
      assert!(difference <= bound, "{difference} > {bound}");
    }
  }

  #[test]
  fn the_factor_is_lower_triangular_with_positive_diagonal()
  {
    let mut lcg = Lcg(7);
    let a = random_spd::<5>(&mut lcg);
    let l = *CholeskyFactorization::factorize(&a).unwrap().lower();
    for i in 0..5
    {
      assert!(l[i][i] > 0.0);
      for j in (i + 1)..5
      {
        assert_eq!(l[i][j], 0.0);
      }
    }
  }

  #[test]
  fn solves_a_system_with_a_known_solution()
  {
    let mut lcg = Lcg(11);
    for _ in 0..20
    {
      let a = random_spd::<7>(&mut lcg);
      let mut x_true = [0.0; 7];
      for value in x_true.iter_mut()
      {
        *value = lcg.standard_normal();
      }
      let mut b = [0.0; 7];
      for i in 0..7
      {
        for j in 0..7
        {
          b[i] += a[i][j] * x_true[j];
        }
      }
      let x = CholeskyFactorization::factorize(&a).unwrap().solve(&b);
      for i in 0..7
      {
        assert!((x[i] - x_true[i]).abs() <= 1.0e-12, "{} vs {}", x[i], x_true[i]);
      }
    }
  }

  #[test]
  fn reports_the_first_non_positive_pivot_instead_of_producing_nan()
  {
    // Symmetric, indefinite: eigenvalues 3 and -1. The first pivot is 1 > 0;
    // the Schur complement is 1 - 2*2/1 = -3 at index 1.
    let a = [[1.0, 2.0], [2.0, 1.0]];
    let error = CholeskyFactorization::factorize(&a).unwrap_err();
    assert_eq!(error.pivot_index, 1);
    let pivot: f64 = error.pivot_value;
    assert!((pivot + 3.0).abs() <= 1.0e-15);

    // Zero matrix fails at the very first pivot.
    let zero = [[0.0; 3]; 3];
    assert_eq!(
      CholeskyFactorization::factorize(&zero).unwrap_err().pivot_index, 0);
  }

  #[test]
  fn log_determinant_matches_a_diagonal_matrix()
  {
    let a = [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 0.25]];
    let expected = (4.0f64 * 9.0 * 0.25).ln();
    let actual = CholeskyFactorization::factorize(&a).unwrap().log_determinant();
    assert!((actual - expected).abs() <= 1.0e-15);
  }

  #[test]
  fn transforming_white_noise_reproduces_the_covariance()
  {
    let p = [[4.0, 1.2], [1.2, 1.0]];
    let factorization = CholeskyFactorization::factorize(&p).unwrap();
    let mut lcg = Lcg(2026);
    let draws = 200_000usize;
    let mut sum = [0.0f64; 2];
    let mut sum_outer = [[0.0f64; 2]; 2];
    for _ in 0..draws
    {
      let z = [lcg.standard_normal(), lcg.standard_normal()];
      let x = factorization.transform(&z);
      for i in 0..2
      {
        sum[i] += x[i];
        for j in 0..2
        {
          sum_outer[i][j] += x[i] * x[j];
        }
      }
    }
    let n = draws as f64;
    for i in 0..2
    {
      for j in 0..2
      {
        let sample_covariance =
          sum_outer[i][j] / n - (sum[i] / n) * (sum[j] / n);
        // Standard error of a covariance estimate is O(1/sqrt(n)) times the
        // scale of the entries; 4 sigma at n = 2e5 is comfortably under 0.05.
        assert!(
          (sample_covariance - p[i][j]).abs() <= 0.05,
          "P[{i}][{j}] = {} vs {}", sample_covariance, p[i][j]);
      }
    }
  }

  #[test]
  fn whitening_gives_the_mahalanobis_norm()
  {
    let p = [[2.0, 0.5], [0.5, 1.0]];
    let factorization = CholeskyFactorization::factorize(&p).unwrap();
    let d = [1.0, -2.0];
    let y = factorization.solve_lower(&d);
    let whitened_norm_squared: f64 = y[0] * y[0] + y[1] * y[1];
    // d^T P^{-1} d computed directly.
    let x = factorization.solve(&d);
    let direct: f64 = d[0] * x[0] + d[1] * x[1];
    assert!((whitened_norm_squared - direct).abs() <= 1.0e-14);
  }
}
