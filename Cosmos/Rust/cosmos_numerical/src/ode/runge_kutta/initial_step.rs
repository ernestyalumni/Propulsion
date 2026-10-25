//! The starting step size of HNW I §II.4, "Starting Step Size", eq. (4.14).
//!
//! # The mathematics
//!
//! The local error of a method of order `p` is about `C h^{p+1} y^{(p+1)}`.
//! The unknown derivative is replaced by estimates of the first and second
//! derivative of the solution:
//!
//! a. `d_0 = ‖y_0‖`, `d_1 = ‖f(x_0, y_0)‖`, both in the scaled norm with
//!    `sc_i = Atol_i + |y_0i| Rtol_i`;
//! b. `h_0 = 0.01 d_0 / d_1`, so an explicit Euler step changes `y_0` by about
//!    one percent (or `h_0 = 10⁻⁶` when `d_0` or `d_1` is below `10⁻⁵`);
//! c. one Euler step `y_1 = y_0 + h_0 f(x_0, y_0)` and one evaluation
//!    `f(x_0 + h_0, y_1)`;
//! d. `d_2 = ‖f(x_0 + h_0, y_1) − f(x_0, y_0)‖ / h_0`;
//! e. `h_1` from `h_1^k · max(d_1, d_2) = 0.01` (or
//!    `h_1 = max(10⁻⁶, 10⁻³ h_0)` when `max(d_1, d_2) ≤ 10⁻¹⁵`);
//! f. `h = min(100 h_0, h_1, h_max)`.
//!
//! The book writes `k = p + 1` in (e). Hairer's codes use `k = IORD`, 5 for
//! DOPRI5 and 8 for DOP853, and take `d_1, d_2` without the `1/n` of the
//! root-mean-square norm (it cancels in `d_0 / d_1` but not in (e)). This
//! module follows the codes, which is what the golden vectors in
//! `golden/hairer_reference/` were produced by.

use super::tolerance::Tolerance;
use crate::field::RealField;

pub const EULER_INCREMENT_FRACTION: f64 = 0.01;
pub const TINY_SQUARED_NORM: f64 = 1.0e-10;
pub const FALLBACK_FIRST_GUESS: f64 = 1.0e-6;
pub const TINY_DERIVATIVE: f64 = 1.0e-15;
pub const FALLBACK_SECOND_GUESS_FLOOR: f64 = 1.0e-6;
pub const FALLBACK_SECOND_GUESS_FACTOR: f64 = 1.0e-3;
pub const MAXIMUM_GROWTH_OVER_FIRST_GUESS: f64 = 100.0;

#[allow(clippy::too_many_arguments)]
/// The proposed first step, signed by `direction` (`+1` forward, `-1`
/// backward). Spends exactly one evaluation of `rhs`.
pub fn initial_step_size<T, const N: usize, F>(
  rhs: &mut F,
  t_0: T,
  y_0: &[T; N],
  f_0: &[T; N],
  direction: T,
  order: u32,
  maximum_step: T,
  tolerance: &Tolerance<T, N>) -> T
where
  T: RealField,
  F: FnMut(T, &[T; N]) -> [T; N] + ?Sized,
{
  let mut squared_norm_of_derivative = T::zero();
  let mut squared_norm_of_state = T::zero();
  for i in 0..N
  {
    let scale = tolerance.scale(i, y_0[i], y_0[i]);
    squared_norm_of_derivative =
      squared_norm_of_derivative + (f_0[i] / scale) * (f_0[i] / scale);
    squared_norm_of_state =
      squared_norm_of_state + (y_0[i] / scale) * (y_0[i] / scale);
  }

  let tiny = T::from_f64(TINY_SQUARED_NORM);
  let mut h = if squared_norm_of_derivative <= tiny
    || squared_norm_of_state <= tiny
  {
    T::from_f64(FALLBACK_FIRST_GUESS)
  }
  else
  {
    (squared_norm_of_state / squared_norm_of_derivative).square_root()
      * T::from_f64(EULER_INCREMENT_FRACTION)
  };
  h = h.minimum(maximum_step);
  h = with_sign_of(h, direction);

  let mut y_euler = [T::zero(); N];
  for i in 0..N
  {
    y_euler[i] = y_0[i] + h * f_0[i];
  }
  let f_euler = rhs(t_0 + h, &y_euler);

  let mut squared_norm_of_difference = T::zero();
  for i in 0..N
  {
    let scale = tolerance.scale(i, y_0[i], y_0[i]);
    let scaled = (f_euler[i] - f_0[i]) / scale;
    squared_norm_of_difference = squared_norm_of_difference + scaled * scaled;
  }
  let second_derivative = squared_norm_of_difference.square_root() / h;
  let largest_derivative = second_derivative
    .absolute_value()
    .maximum(squared_norm_of_derivative.square_root());

  let second_guess = if largest_derivative <= T::from_f64(TINY_DERIVATIVE)
  {
    T::from_f64(FALLBACK_SECOND_GUESS_FLOOR)
      .maximum(h.absolute_value() * T::from_f64(FALLBACK_SECOND_GUESS_FACTOR))
  }
  else
  {
    (T::from_f64(EULER_INCREMENT_FRACTION) / largest_derivative)
      .power(T::one() / T::from_f64(order as f64))
  };

  let proposed = (T::from_f64(MAXIMUM_GROWTH_OVER_FIRST_GUESS)
    * h.absolute_value())
    .minimum(second_guess)
    .minimum(maximum_step);
  with_sign_of(proposed, direction)
}

/// Fortran's `SIGN(a, b)`: `|a|` carrying the sign of `b`.
pub(crate) fn with_sign_of<T: RealField>(magnitude: T, sign: T) -> T
{
  if sign < T::zero() { -magnitude.absolute_value() }
  else { magnitude.absolute_value() }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn spends_one_evaluation_and_respects_the_maximum_step()
  {
    let mut evaluations = 0;
    let mut rhs = |_t: f64, y: &[f64; 1]| {
      evaluations += 1;
      [-y[0]]
    };
    let tolerance = Tolerance::uniform(1.0e-6, 1.0e-6).unwrap();
    let h = initial_step_size(
      &mut rhs, 0.0, &[1.0], &[-1.0], 1.0, 5, 1.0e-3, &tolerance);
    assert_eq!(evaluations, 1);
    assert!(h > 0.0 && h <= 1.0e-3, "{h}");
  }

  #[test]
  fn backward_integration_gets_a_negative_step()
  {
    let mut rhs = |_t: f64, y: &[f64; 1]| [-y[0]];
    let tolerance = Tolerance::uniform(1.0e-6, 1.0e-6).unwrap();
    let h = initial_step_size(
      &mut rhs, 0.0, &[1.0], &[-1.0], -1.0, 5, 10.0, &tolerance);
    assert!(h < 0.0, "{h}");
  }

  /// For y' = -y the estimate is analytic: d_1 = d_2 = ‖y_0‖, so
  /// h = min(100 · 0.01, (0.01 / d_1)^{1/k}).
  #[test]
  fn reproduces_the_closed_form_for_exponential_decay()
  {
    let mut rhs = |_t: f64, y: &[f64; 1]| [-y[0]];
    let tolerance = Tolerance::uniform(1.0e-8, 0.0).unwrap();
    let h = initial_step_size(
      &mut rhs, 0.0, &[1.0], &[-1.0], 1.0, 5, 100.0, &tolerance);
    let d_1: f64 = 1.0 / 1.0e-8;
    let expected = (100.0_f64 * 0.01).min((0.01 / d_1).powf(0.2));
    assert!((h - expected).abs() <= 1.0e-12 * expected, "{h} vs {expected}");
  }
}
