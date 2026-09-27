//! Classical fourth-order Runge-Kutta on a fixed-size state, with the tableau
//! named, and a step that lands on an event (burnout) instead of crossing it.
//!
//! RK4 preserves every linear invariant l . y with l . f == 0 to round-off,
//! which the mass-conservation tests rely on (derivation note, section 7b).
//! C++ twin: `Propulsion/SolidRocketMotor/RungeKutta4.h`.

use cosmos_numerical::field::RealField;

pub struct RungeKutta4Tableau;

impl RungeKutta4Tableau
{
  pub const C2: f64 = 0.5;
  pub const C3: f64 = 0.5;
  pub const C4: f64 = 1.0;
  pub const A21: f64 = 0.5;
  pub const A32: f64 = 0.5;
  pub const A43: f64 = 1.0;
  pub const B1: f64 = 1.0 / 6.0;
  pub const B2: f64 = 1.0 / 3.0;
  pub const B3: f64 = 1.0 / 3.0;
  pub const B4: f64 = 1.0 / 6.0;
}

/// 64 halvings take any double step below its unit round-off.
pub const EVENT_BISECTION_ITERATIONS: usize = 64;

pub fn runge_kutta_4_step<T: RealField, const N: usize, F>(f: &F, t: T, y: &[T; N], h: T) -> [T; N]
where
  F: Fn(T, &[T; N]) -> [T; N],
{
  let c2 = T::from_f64(RungeKutta4Tableau::C2);
  let c3 = T::from_f64(RungeKutta4Tableau::C3);
  let c4 = T::from_f64(RungeKutta4Tableau::C4);
  let a21 = T::from_f64(RungeKutta4Tableau::A21);
  let a32 = T::from_f64(RungeKutta4Tableau::A32);
  let a43 = T::from_f64(RungeKutta4Tableau::A43);
  let b1 = T::from_f64(RungeKutta4Tableau::B1);
  let b2 = T::from_f64(RungeKutta4Tableau::B2);
  let b3 = T::from_f64(RungeKutta4Tableau::B3);
  let b4 = T::from_f64(RungeKutta4Tableau::B4);

  let mut stage = [T::zero(); N];
  let k1 = f(t, y);
  for i in 0..N
  {
    stage[i] = y[i] + h * a21 * k1[i];
  }
  let k2 = f(t + c2 * h, &stage);
  for i in 0..N
  {
    stage[i] = y[i] + h * a32 * k2[i];
  }
  let k3 = f(t + c3 * h, &stage);
  for i in 0..N
  {
    stage[i] = y[i] + h * a43 * k3[i];
  }
  let k4 = f(t + c4 * h, &stage);

  let mut next = [T::zero(); N];
  for i in 0..N
  {
    next[i] = y[i] + h * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i] + b4 * k4[i]);
  }
  next
}

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

  /// y' = -y: RK4 error at t = 1 falls by about 2^4 per halving.
  #[test]
  fn converges_at_fourth_order()
  {
    let f = |_t: f64, y: &[f64; 1]| [-y[0]];
    let error = |steps: usize| {
      let h = 1.0 / steps as f64;
      let mut y = [1.0];
      for n in 0..steps
      {
        y = runge_kutta_4_step(&f, n as f64 * h, &y, h);
      }
      (y[0] - (-1.0_f64).exp()).abs()
    };
    let order = (error(10) / error(20)).log2();
    assert!((order - 4.0).abs() < 0.1, "observed order {order}");
  }

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
