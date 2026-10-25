//! The error scale and the error norm of HNW I §II.4.
//!
//! # The mathematics
//!
//! A step from `y_0` to `y_1` is judged component by component against
//!
//! `sc_i = Atol_i + max(|y_0i|, |y_1i|) · Rtol_i`          (HNW I eq. (4.10)),
//!
//! and the error estimate `δ` of an embedded pair is reduced to one number by
//! the root-mean-square norm
//!
//! `err = sqrt( (1/n) Σ_i (δ_i / sc_i)² )`                   (HNW I eq. (4.11)).
//!
//! `err ≤ 1` accepts the step. Each `Atol_i` is required to be positive, so
//! `sc_i > 0` for every state, including a component that passes through zero
//! under a purely relative tolerance; `Rtol_i` may be zero.

use crate::field::RealField;

/// A tolerance outside its domain, naming the offending component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToleranceError
{
  NonPositiveAbsoluteTolerance { component: usize },
  NegativeRelativeTolerance { component: usize },
  NonFiniteTolerance { component: usize },
}

/// Absolute and relative tolerances for each of the `N` components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance<T: RealField, const N: usize>
{
  absolute: [T; N],
  relative: [T; N],
}

impl<T: RealField, const N: usize> Tolerance<T, N>
{
  /// One absolute and one relative tolerance for every component (Hairer's
  /// `ITOL = 0`).
  pub fn uniform(absolute: T, relative: T) -> Result<Self, ToleranceError>
  {
    Self::componentwise([absolute; N], [relative; N])
  }

  /// A tolerance pair per component (Hairer's `ITOL = 1`).
  pub fn componentwise(
    absolute: [T; N],
    relative: [T; N]) -> Result<Self, ToleranceError>
  {
    for component in 0..N
    {
      if !absolute[component].is_finite() || !relative[component].is_finite()
      {
        return Err(ToleranceError::NonFiniteTolerance { component });
      }
      if absolute[component] <= T::zero()
      {
        return Err(ToleranceError::NonPositiveAbsoluteTolerance { component });
      }
      if relative[component] < T::zero()
      {
        return Err(ToleranceError::NegativeRelativeTolerance { component });
      }
    }
    Ok(Self { absolute, relative })
  }

  pub fn absolute(&self) -> &[T; N] { &self.absolute }
  pub fn relative(&self) -> &[T; N] { &self.relative }

  /// `sc_i` of eq. (4.10) for the step from `y_0` to `y_1`.
  pub fn scale(&self, component: usize, y_0: T, y_1: T) -> T
  {
    self.absolute[component]
      + self.relative[component]
        * y_0.absolute_value().maximum(y_1.absolute_value())
  }

  /// `Σ_i (δ_i / sc_i)²`, the sum inside eq. (4.11), before the mean and the
  /// square root. Methods with a composite estimator (DOP853) combine two of
  /// these sums, so the sum is the shared primitive.
  pub fn sum_of_squared_scaled(
    &self,
    error: &[T; N],
    y_0: &[T; N],
    y_1: &[T; N]) -> T
  {
    let mut sum = T::zero();
    for component in 0..N
    {
      let scaled =
        error[component] / self.scale(component, y_0[component], y_1[component]);
      sum = sum + scaled * scaled;
    }
    sum
  }

  /// `err` of eq. (4.11).
  pub fn error_norm(&self, error: &[T; N], y_0: &[T; N], y_1: &[T; N]) -> T
  {
    (self.sum_of_squared_scaled(error, y_0, y_1) / T::from_f64(N as f64))
      .square_root()
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn rejects_tolerances_outside_their_domain()
  {
    use ToleranceError::*;
    assert_eq!(
      Tolerance::<f64, 2>::componentwise([1.0e-6, 0.0], [1.0e-6, 1.0e-6])
        .unwrap_err(),
      NonPositiveAbsoluteTolerance { component: 1 });
    assert_eq!(
      Tolerance::<f64, 2>::componentwise([1.0e-6, 1.0e-6], [-1.0, 0.0])
        .unwrap_err(),
      NegativeRelativeTolerance { component: 0 });
    assert_eq!(
      Tolerance::<f64, 1>::uniform(f64::NAN, 0.0).unwrap_err(),
      NonFiniteTolerance { component: 0 });
    assert!(Tolerance::<f64, 3>::uniform(1.0e-9, 0.0).is_ok());
  }

  #[test]
  fn an_error_exactly_at_the_scale_has_unit_norm()
  {
    let tolerance =
      Tolerance::<f64, 2>::componentwise([1.0e-3, 2.0e-3], [1.0e-2, 0.0])
        .unwrap();
    let y_0 = [2.0, -5.0];
    let y_1 = [-3.0, 1.0];
    let error = [1.0e-3 + 3.0e-2, 2.0e-3];
    assert!((tolerance.error_norm(&error, &y_0, &y_1) - 1.0).abs() < 1.0e-15);
  }

  #[test]
  fn the_norm_is_the_root_mean_square_of_the_scaled_components()
  {
    let tolerance = Tolerance::<f64, 4>::uniform(1.0, 0.0).unwrap();
    let zero = [0.0; 4];
    let error = [1.0, 1.0, 1.0, 5.0];
    assert!((tolerance.error_norm(&error, &zero, &zero) - 7.0_f64.sqrt()).abs()
      < 1.0e-15);
  }
}
