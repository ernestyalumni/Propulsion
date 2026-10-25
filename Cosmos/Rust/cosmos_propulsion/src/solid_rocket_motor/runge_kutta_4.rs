//! A classical RK4 step that lands on an event (burnout) instead of crossing
//! it. The step and its tableau live in `cosmos_numerical`; they are
//! re-exported here so the solid-rocket-motor code keeps one import path.
//!
//! RK4 preserves every linear invariant l . y with l . f == 0 to round-off,
//! which the mass-conservation tests rely on (derivation note, section 7b).
//! C++ twin: `Propulsion/SolidRocketMotor/RungeKutta4.h`.

use cosmos_numerical::field::RealField;
pub use cosmos_numerical::ode::runge_kutta::classical_4::{
  runge_kutta_4_step, RungeKutta4Tableau,
};

/// 64 halvings take any double step below its unit round-off.
pub const EVENT_BISECTION_ITERATIONS: usize = 64;

/// A step of size h that stops at the moment y[component] reaches threshold,
/// sets it exactly, and finishes the remaining h - s.
pub fn runge_kutta_4_step_stopping_at<T: RealField, const N: usize, F>(
  f: &F,
  t: T,
  y: &[T; N],
  h: T,
  component: usize,
  threshold: T,
) -> [T; N]
where
  F: Fn(T, &[T; N]) -> [T; N],
{
  let full = runge_kutta_4_step(f, t, y, h);
  if !(y[component] < threshold && full[component] > threshold)
  {
    return full;
  }
  let two = T::from_f64(2.0);
  let mut lower = T::zero();
  let mut upper = h;
  for _ in 0..EVENT_BISECTION_ITERATIONS
  {
    let middle = (lower + upper) / two;
    if runge_kutta_4_step(f, t, y, middle)[component] < threshold
    {
      lower = middle;
    }
    else
    {
      upper = middle;
    }
  }
  let mut landed = runge_kutta_4_step(f, t, y, upper);
  landed[component] = threshold;
  runge_kutta_4_step(f, t + upper, &landed, h - upper)
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn lands_on_the_event()
  {
    let f = |_t: f64, _y: &[f64; 2]| [1.0, 2.0];
    let next = runge_kutta_4_step_stopping_at(&f, 0.0, &[0.0, 0.0], 1.0, 0, 0.3);
    assert!((next[1] - 2.0).abs() < 1.0e-15);
    let before = runge_kutta_4_step_stopping_at(&f, 0.0, &[0.0, 0.0], 0.2, 0, 0.3);
    assert_eq!(before, runge_kutta_4_step(&f, 0.0, &[0.0, 0.0], 0.2));
  }
}
