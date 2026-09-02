//! Proportional–integral step-size control for embedded Runge–Kutta pairs.
//!
//! # The mathematics
//!
//! An embedded pair returns, with each step of size `h`, a scaled error
//! estimate `err` normalized so that `err <= 1` means the step is acceptable.
//! For a method of order `q` the local error scales as `h^(q+1)`, so the step
//! that would have produced exactly `err = 1` is `h · err^(-1/(q+1))`. That is
//! the proportional (I-controller) law of NR §17.2 and HNW I §II.4:
//!
//! `h_new = h · safety · err^(-alpha)`,  `alpha ≈ 1/(q+1)`.
//!
//! Gustafsson's proportional–integral law adds memory of the previous step's
//! error so that the step sequence does not oscillate on mildly stiff problems
//! (HNW II §IV.2, "Lund stabilization"):
//!
//! `h_new = h · safety · err^(-alpha) · err_previous^(beta)`,
//!
//! with the ratio `h_new / h` clamped to `[min_scale, max_scale]`, and never
//! allowed above one in the step immediately after a rejection.
//!
//! Setting `beta = 0` recovers the plain I-controller. NR's shipped code does
//! exactly that as a buried literal; here `beta` is a constructor parameter so
//! the choice is visible and testable.
//!
//! # Twin
//!
//! `Cosmos/Source/Numerical/ODE/RKMethods/ComputePIStepSize.h`. The golden
//! vectors in `golden/pi_step_size.tsv` are emitted from that header and the
//! test below asserts agreement to 1e-15 relative.

use crate::field::RealField;

/// A parameter outside its mathematical domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PIStepSizeControlError
{
  NegativeAlpha,
  NegativeBeta,
  NonPositiveMinScale,
  MaxScaleBelowMinScale,
  NonPositiveSafetyFactor,
}

/// The PI step-size law with every constant named.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PIStepSizeControl<T: RealField>
{
  alpha: T,
  beta: T,
  min_scale: T,
  max_scale: T,
  safety_factor: T,
}

impl<T: RealField> PIStepSizeControl<T>
{
  /// HNW I p. 168 chooses `facmin` between 0.2 and 0.5 and `facmax` between
  /// 1.5 and 5; the C++ twin defaults to 0.2, 5.0, 0.9.
  pub const DEFAULT_MIN_SCALE: f64 = 0.2;
  pub const DEFAULT_MAX_SCALE: f64 = 5.0;
  pub const DEFAULT_SAFETY_FACTOR: f64 = 0.9;

  pub fn new(
    alpha: T,
    beta: T,
    min_scale: T,
    max_scale: T,
    safety_factor: T) -> Result<Self, PIStepSizeControlError>
  {
    if alpha < T::zero()
    {
      return Err(PIStepSizeControlError::NegativeAlpha);
    }
    if beta < T::zero()
    {
      return Err(PIStepSizeControlError::NegativeBeta);
    }
    if min_scale <= T::zero()
    {
      return Err(PIStepSizeControlError::NonPositiveMinScale);
    }
    if max_scale < min_scale
    {
      return Err(PIStepSizeControlError::MaxScaleBelowMinScale);
    }
    if safety_factor <= T::zero()
    {
      return Err(PIStepSizeControlError::NonPositiveSafetyFactor);
    }

    Ok(Self { alpha, beta, min_scale, max_scale, safety_factor })
  }

  pub fn with_default_bounds(
    alpha: T,
    beta: T) -> Result<Self, PIStepSizeControlError>
  {
    Self::new(
      alpha,
      beta,
      T::from_f64(Self::DEFAULT_MIN_SCALE),
      T::from_f64(Self::DEFAULT_MAX_SCALE),
      T::from_f64(Self::DEFAULT_SAFETY_FACTOR))
  }

  pub fn alpha(&self) -> T { self.alpha }
  pub fn beta(&self) -> T { self.beta }
  pub fn min_scale(&self) -> T { self.min_scale }
  pub fn max_scale(&self) -> T { self.max_scale }
  pub fn safety_factor(&self) -> T { self.safety_factor }

  /// The next step size given the scaled error of the step just taken.
  ///
  /// `error <= 1` means the step was accepted; the new step may grow, bounded
  /// by `max_scale`, or by one if the previous attempt was rejected. An error
  /// of exactly zero means the largest allowed growth. `error > 1` means the
  /// step was rejected; the new step shrinks by at least `min_scale` and the
  /// integral term is not applied, matching the C++ twin.
  pub fn compute_new_step_size(
    &self,
    error: T,
    previous_error: T,
    h: T,
    was_rejected: bool) -> T
  {
    if error <= T::one()
    {
      let unclamped = if error == T::zero()
      {
        self.max_scale
      }
      else
      {
        self.safety_factor
          * error.power(-self.alpha)
          * previous_error.power(self.beta)
      };

      let scale = unclamped.maximum(self.min_scale).minimum(self.max_scale);

      return if was_rejected
      {
        h * scale.minimum(T::one())
      }
      else
      {
        h * scale
      };
    }

    let scale =
      (self.safety_factor * error.power(-self.alpha)).maximum(self.min_scale);
    h * scale
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  const GOLDEN: &str = include_str!("../../../../golden/pi_step_size.tsv");

  fn relative_difference(a: f64, b: f64) -> f64
  {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 { (a - b).abs() } else { (a - b).abs() / scale }
  }

  #[test]
  fn agrees_with_the_cpp_twin_on_every_golden_vector()
  {
    let mut checked = 0usize;
    for line in GOLDEN.lines()
    {
      if line.starts_with('#') || line.starts_with("alpha")
      {
        continue;
      }
      let fields: Vec<&str> = line.split('\t').collect();
      assert_eq!(fields.len(), 10, "malformed golden line: {line}");
      let value = |i: usize| fields[i].parse::<f64>().unwrap();

      let controller = PIStepSizeControl::new(
        value(0), value(1), value(2), value(3), value(4)).unwrap();
      let was_rejected = fields[8] == "1";
      let expected = value(9);
      let actual = controller.compute_new_step_size(
        value(5), value(6), value(7), was_rejected);

      assert!(
        relative_difference(actual, expected) <= 1.0e-15,
        "mismatch on {line}: rust {actual:.17e} vs c++ {expected:.17e}");
      checked += 1;
    }
    assert!(checked >= 1000, "golden file looks truncated: {checked} rows");
  }

  #[test]
  fn rejects_parameters_outside_their_domain()
  {
    use PIStepSizeControlError::*;
    assert_eq!(
      PIStepSizeControl::new(-0.1, 0.0, 0.2, 5.0, 0.9).unwrap_err(),
      NegativeAlpha);
    assert_eq!(
      PIStepSizeControl::new(0.2, -0.1, 0.2, 5.0, 0.9).unwrap_err(),
      NegativeBeta);
    assert_eq!(
      PIStepSizeControl::new(0.2, 0.0, 0.0, 5.0, 0.9).unwrap_err(),
      NonPositiveMinScale);
    assert_eq!(
      PIStepSizeControl::new(0.2, 0.0, 0.5, 0.2, 0.9).unwrap_err(),
      MaxScaleBelowMinScale);
    assert_eq!(
      PIStepSizeControl::new(0.2, 0.0, 0.2, 5.0, 0.0).unwrap_err(),
      NonPositiveSafetyFactor);
  }

  #[test]
  fn accepted_step_ratio_stays_inside_the_declared_bounds()
  {
    let controller =
      PIStepSizeControl::with_default_bounds(0.7 / 5.0, 0.08).unwrap();
    for error in [0.0, 1.0e-12, 1.0e-3, 0.5, 1.0]
    {
      for previous_error in [1.0e-9, 0.1, 1.0, 9.0]
      {
        let ratio =
          controller.compute_new_step_size(error, previous_error, 1.0, false);
        assert!(ratio >= 0.2 - 1.0e-15 && ratio <= 5.0 + 1.0e-15, "{ratio}");
      }
    }
  }

  #[test]
  fn a_step_after_a_rejection_never_grows()
  {
    let controller =
      PIStepSizeControl::with_default_bounds(0.7 / 5.0, 0.08).unwrap();
    for error in [0.0, 1.0e-12, 0.3, 1.0]
    {
      let h_new = controller.compute_new_step_size(error, 0.5, 2.0, true);
      assert!(h_new <= 2.0 + 1.0e-15, "{h_new}");
    }
  }

  #[test]
  fn a_rejected_step_shrinks_and_shrinks_more_for_larger_error()
  {
    let controller =
      PIStepSizeControl::with_default_bounds(0.7 / 5.0, 0.08).unwrap();
    let mut previous = f64::INFINITY;
    for error in [1.5, 4.0, 100.0, 1.0e6]
    {
      let h_new = controller.compute_new_step_size(error, 0.5, 1.0, false);
      assert!(h_new < 1.0);
      assert!(h_new <= previous);
      previous = h_new;
    }
    assert!((previous - 0.2).abs() <= 1.0e-15, "floor at min_scale: {previous}");
  }

  #[test]
  fn zero_beta_is_the_plain_i_controller()
  {
    let pi = PIStepSizeControl::with_default_bounds(0.2, 0.0).unwrap();
    for previous_error in [1.0e-9, 0.3, 4.0]
    {
      let a = pi.compute_new_step_size(0.1, previous_error, 1.0, false);
      let b = pi.compute_new_step_size(0.1, 1.0, 1.0, false);
      assert_eq!(a, b);
    }
  }
}
