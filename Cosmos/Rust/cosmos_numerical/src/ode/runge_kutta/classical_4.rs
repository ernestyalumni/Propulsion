//! Classical fourth-order Runge–Kutta on a fixed-size state, with the tableau
//! named (HNW I §II.1, Table 1.2, left).
//!
//! ```text
//!   0   |
//!  1/2  | 1/2
//!  1/2  |  0   1/2
//!   1   |  0    0    1
//! ------+--------------------
//!       | 1/6  1/3  1/3  1/6
//! ```
//!
//! RK4 preserves every linear invariant l . y with l . f == 0 to round-off.
//! The arithmetic below is kept in the operation order of the C++ twin
//! `Propulsion/SolidRocketMotor/RungeKutta4.h`, so trajectories built on it
//! agree bitwise with C++ (the solid-rocket-motor golden vectors rely on this).

use crate::field::RealField;

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

/// One step y(t) -> y(t + h) of classical RK4 for y' = f(t, y).
pub fn runge_kutta_4_step<T: RealField, const N: usize, F>(
  f: &F,
  t: T,
  y: &[T; N],
  h: T) -> [T; N]
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
}
