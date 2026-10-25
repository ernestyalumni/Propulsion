//! DOPRI5: the Dormand–Prince 5(4) pair with its order-4 continuous
//! extension, HNW I §II.5 Table 5.2 and §II.6 (Hairer's `DOPRI5`, 1996).
//!
//! # The method
//!
//! Seven stages; the seventh is evaluated at `(t + h, y_1)`, so it is the
//! first stage of the next step (first same as last, six evaluations per
//! step). `y_1 = y_0 + h Σ b_j k_j` is of order 5 and is the one kept (local
//! extrapolation); `ŷ_1 = y_0 + h Σ b̂_j k_j` of order 4 only measures the
//! error, `y_1 − ŷ_1 = h Σ e_j k_j` with `e = b − b̂`. The step exponent is
//! `1/(q+1) = 1/5`, `q = min(5, 4)`.
//!
//! # Dense output
//!
//! With `Δ = y_1 − y_0`, `β = h k_1 − Δ`, and the stored vectors
//! `r_0 = y_0, r_1 = Δ, r_2 = β, r_3 = Δ − h k_7 − β, r_4 = h Σ d_j k_j`,
//!
//! `u(θ) = r_0 + θ (r_1 + (1−θ) (r_2 + θ (r_3 + (1−θ) r_4)))`,
//!
//! a quintic in `θ` that matches `y_0, f_0` at `θ = 0` and `y_1, f_1` at
//! `θ = 1` and is accurate to order 4 in between (`CONTD5`).
//!
//! # Stiffness
//!
//! Stages 6 and 7 share the abscissa `t + h`, so
//! `h|λ| ≈ |h| ‖k_7 − k_6‖ / ‖y_1 − Y_6‖`; the stability region of DOPRI5
//! reaches about `−3.3` on the real axis, and the test fires above `3.25`.

use super::embedded::{
  stage_input, stiffness_quotient, weighted_sum, EmbeddedPair,
  EmbeddedRungeKutta, IntegrationError, IntegratorOptions, Interpolant, Trial,
};
use super::tolerance::Tolerance;
use crate::field::RealField;

pub const STAGES: usize = 7;

/// The Butcher tableau. Row 7 of `A` is `b`: the seventh stage is
/// `f(t + h, y_1)`.
pub struct Tableau;

impl Tableau
{
  pub const C: [f64; STAGES] = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];

  pub const A: [[f64; STAGES]; STAGES] = [
    [0.0; STAGES],
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
    [
      19372.0 / 6561.0,
      -25360.0 / 2187.0,
      64448.0 / 6561.0,
      -212.0 / 729.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      9017.0 / 3168.0,
      -355.0 / 33.0,
      46732.0 / 5247.0,
      49.0 / 176.0,
      -5103.0 / 18656.0,
      0.0,
      0.0,
    ],
    Self::B,
  ];

  pub const B: [f64; STAGES] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
  ];

  /// `e = b − b̂`.
  pub const E: [f64; STAGES] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
  ];

  /// The coefficients of `r_4` in the continuous extension.
  pub const D: [f64; STAGES] = [
    -12715105075.0 / 11282082432.0,
    0.0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0,
  ];
}

/// The stage storage of one step.
#[derive(Clone, Debug)]
pub struct DormandPrince5<T: RealField, const N: usize>
{
  stages: [[T; N]; STAGES],
  sixth_stage_input: [T; N],
}

impl<T: RealField, const N: usize> Default for DormandPrince5<T, N>
{
  fn default() -> Self
  {
    Self {
      stages: [[T::zero(); N]; STAGES],
      sixth_stage_input: [T::zero(); N],
    }
  }
}

/// `u(θ)` of one accepted DOPRI5 step.
#[derive(Clone, Debug)]
pub struct DormandPrince5Interpolant<T: RealField, const N: usize>
{
  coefficients: [[T; N]; 5],
}

impl<T: RealField, const N: usize> Interpolant<T, N>
  for DormandPrince5Interpolant<T, N>
{
  fn evaluate(&self, theta: T) -> [T; N]
  {
    let one_minus_theta = T::one() - theta;
    let r = &self.coefficients;
    let mut value = [T::zero(); N];
    for i in 0..N
    {
      value[i] = r[0][i]
        + theta
          * (r[1][i]
            + one_minus_theta
              * (r[2][i] + theta * (r[3][i] + one_minus_theta * r[4][i])));
    }
    value
  }
}

impl<T: RealField, const N: usize> EmbeddedPair<T, N> for DormandPrince5<T, N>
{
  const NAME: &'static str = "DOPRI5";
  const STARTING_STEP_ORDER: u32 = 5;
  const ERROR_EXPONENT: f64 = 0.2;
  const BETA_CORRECTION: f64 = 0.75;
  const DEFAULT_BETA: f64 = 0.04;
  const DEFAULT_MIN_SCALE: f64 = 0.2;
  const DEFAULT_MAX_SCALE: f64 = 10.0;
  const STIFFNESS_LIMIT: f64 = 3.25;
  const EVALUATIONS_PER_ATTEMPT: usize = 6;
  const EVALUATIONS_AT_ACCEPTANCE: usize = 0;
  const EVALUATIONS_FOR_INTERPOLANT: usize = 0;

  type Interpolant = DormandPrince5Interpolant<T, N>;

  fn attempt<F>(
    &mut self,
    rhs: &mut F,
    t: T,
    y: &[T; N],
    f_0: &[T; N],
    h: T,
    tolerance: &Tolerance<T, N>) -> Trial<T, N>
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized,
  {
    self.stages[0] = *f_0;
    let mut y_new = *y;
    for stage in 1..STAGES
    {
      let input = stage_input(y, h, &Tableau::A[stage], &self.stages, stage);
      self.stages[stage] = rhs(t + T::from_f64(Tableau::C[stage]) * h, &input);
      match stage
      {
        5 => self.sixth_stage_input = input,
        6 => y_new = input,
        _ => {}
      }
    }

    let difference = weighted_sum(&Tableau::E, &self.stages, STAGES);
    let mut error_estimate = [T::zero(); N];
    for i in 0..N
    {
      error_estimate[i] = difference[i] * h;
    }
    let error = tolerance.error_norm(&error_estimate, y, &y_new);
    Trial { y_new, error }
  }

  fn derivative_at_end<F>(
    &mut self,
    _rhs: &mut F,
    _t: T,
    _h: T,
    _y_new: &[T; N]) -> [T; N]
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized,
  {
    self.stages[STAGES - 1]
  }

  fn stiffness_estimate(&self, h: T, y_new: &[T; N]) -> Option<T>
  {
    stiffness_quotient(
      h, &self.stages[6], &self.stages[5], y_new, &self.sixth_stage_input)
  }

  fn interpolant<F>(
    &mut self,
    _rhs: &mut F,
    _t: T,
    y: &[T; N],
    h: T,
    y_new: &[T; N]) -> Self::Interpolant
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized,
  {
    let d_sum = weighted_sum(&Tableau::D, &self.stages, STAGES);
    let mut coefficients = [[T::zero(); N]; 5];
    for i in 0..N
    {
      let increment = y_new[i] - y[i];
      let b_spline = h * self.stages[0][i] - increment;
      coefficients[0][i] = y[i];
      coefficients[1][i] = increment;
      coefficients[2][i] = b_spline;
      coefficients[3][i] = (-(h * self.stages[6][i]) + increment) - b_spline;
      coefficients[4][i] = h * d_sum[i];
    }
    DormandPrince5Interpolant { coefficients }
  }
}

/// A DOPRI5 integration of `y' = rhs(t, y)` from `(t_0, y_0)` to `end_time`.
pub fn integrator<T, const N: usize, F>(
  rhs: F,
  t_0: T,
  y_0: [T; N],
  end_time: T,
  tolerance: Tolerance<T, N>,
  options: IntegratorOptions<T>)
  -> Result<EmbeddedRungeKutta<T, N, DormandPrince5<T, N>, F>, IntegrationError<T>>
where
  T: RealField,
  F: FnMut(T, &[T; N]) -> [T; N],
{
  EmbeddedRungeKutta::new(
    DormandPrince5::default(), rhs, t_0, y_0, end_time, tolerance, options)
}

#[cfg(test)]
mod tests
{
  use super::*;
  use crate::ode::runge_kutta::order_conditions::{
    residuals_by_order, row_sum_defect,
  };
  use crate::ode::runge_kutta::test_problems::{
    arenstorf, ARENSTORF_INITIAL_STATE, ARENSTORF_PERIOD,
  };

  fn b_hat() -> [f64; STAGES]
  {
    let mut b_hat = [0.0; STAGES];
    for j in 0..STAGES
    {
      b_hat[j] = Tableau::B[j] - Tableau::E[j];
    }
    b_hat
  }

  #[test]
  fn the_tableau_is_consistent()
  {
    assert!(row_sum_defect(&Tableau::A, &Tableau::C) < 1.0e-15);
    assert_eq!(Tableau::A[6], Tableau::B);
  }

  #[test]
  fn the_propagated_solution_has_exact_order_five()
  {
    let residuals = residuals_by_order(&Tableau::A, &Tableau::B, 6);
    assert!(residuals[..5].iter().all(|r| *r < 1.0e-15), "{residuals:?}");
    assert!(residuals[5] > 1.0e-5, "{residuals:?}");
  }

  #[test]
  fn the_embedded_solution_has_exact_order_four()
  {
    let residuals = residuals_by_order(&Tableau::A, &b_hat(), 5);
    assert!(residuals[..4].iter().all(|r| *r < 1.0e-15), "{residuals:?}");
    assert!(residuals[4] > 1.0e-5, "{residuals:?}");
  }

  /// y' = y cos t, y = exp(sin t). One step's local error falls by 2^6 per
  /// halving (order 5); the dense output at θ = 1/2 by 2^5 (order 4).
  #[test]
  fn local_errors_of_the_step_and_of_the_dense_output()
  {
    let exact = |t: f64| t.sin().exp();
    let mut rhs = |t: f64, y: &[f64; 1]| [y[0] * t.cos()];
    let tolerance = Tolerance::uniform(1.0, 0.0).unwrap();
    let errors = |h: f64, rhs: &mut dyn FnMut(f64, &[f64; 1]) -> [f64; 1]| {
      let mut method = DormandPrince5::<f64, 1>::default();
      let t = 0.3;
      let y = [exact(t)];
      let f_0 = rhs(t, &y);
      let trial = method.attempt(rhs, t, &y, &f_0, h, &tolerance);
      let interpolant = method.interpolant(rhs, t, &y, h, &trial.y_new);
      (
        (trial.y_new[0] - exact(t + h)).abs(),
        (interpolant.evaluate(0.5)[0] - exact(t + 0.5 * h)).abs(),
      )
    };
    let (step_coarse, dense_coarse) = errors(0.2, &mut rhs);
    let (step_fine, dense_fine) = errors(0.1, &mut rhs);
    let step_order = (step_coarse / step_fine).log2() - 1.0;
    let dense_order = (dense_coarse / dense_fine).log2() - 1.0;
    assert!((step_order - 5.0).abs() < 0.3, "step order {step_order}");
    assert!((dense_order - 4.0).abs() < 0.3, "dense order {dense_order}");
  }

  #[test]
  fn the_interpolant_matches_both_ends_of_the_step()
  {
    let mut rhs = |t: f64, y: &[f64; 2]| [y[1], -y[0] + t.sin()];
    let tolerance = Tolerance::uniform(1.0, 0.0).unwrap();
    let mut method = DormandPrince5::<f64, 2>::default();
    let y = [0.4, -1.3];
    let f_0 = rhs(0.0, &y);
    let trial = method.attempt(&mut rhs, 0.0, &y, &f_0, 0.25, &tolerance);
    let interpolant = method.interpolant(&mut rhs, 0.0, &y, 0.25, &trial.y_new);
    for i in 0..2
    {
      assert_eq!(interpolant.evaluate(0.0)[i], y[i]);
      assert!((interpolant.evaluate(1.0)[i] - trial.y_new[i]).abs() < 1.0e-15);
    }
  }

  /// One period of the Arenstorf orbit returns to the initial state; the
  /// closing error falls with the tolerance.
  #[test]
  fn closes_the_arenstorf_orbit()
  {
    let mut previous = f64::INFINITY;
    for tolerance in [1.0e-6, 1.0e-9, 1.0e-12]
    {
      let mut integration = integrator(
        arenstorf,
        0.0,
        ARENSTORF_INITIAL_STATE,
        ARENSTORF_PERIOD,
        Tolerance::uniform(tolerance, tolerance).unwrap(),
        IntegratorOptions::default())
      .unwrap();
      let end = integration.integrate_to_end().unwrap();
      assert_eq!(integration.time(), ARENSTORF_PERIOD);
      let closing = (0..2)
        .map(|i| (end[i] - ARENSTORF_INITIAL_STATE[i]).abs())
        .fold(0.0, f64::max);
      assert!(closing < previous, "{tolerance}: {closing}");
      assert!(closing < 1.0e4 * tolerance, "{tolerance}: {closing}");
      previous = closing;
    }
  }

  #[test]
  fn integrates_backward_in_time()
  {
    let mut integration = integrator(
      arenstorf,
      ARENSTORF_PERIOD,
      ARENSTORF_INITIAL_STATE,
      0.0,
      Tolerance::uniform(1.0e-10, 1.0e-10).unwrap(),
      IntegratorOptions::default())
    .unwrap();
    let end = integration.integrate_to_end().unwrap();
    assert_eq!(integration.time(), 0.0);
    assert!((end[0] - ARENSTORF_INITIAL_STATE[0]).abs() < 1.0e-5, "{end:?}");
  }

  /// The dense output of a real-time loop: frames at 60 Hz on exp decay.
  #[test]
  fn solution_at_frame_times_tracks_the_exact_solution()
  {
    let mut integration = integrator(
      |_t: f64, y: &[f64; 1]| [-y[0]],
      0.0,
      [1.0],
      10.0,
      Tolerance::uniform(1.0e-10, 1.0e-10).unwrap(),
      IntegratorOptions::default())
    .unwrap();
    for frame in 0..=600
    {
      let t = frame as f64 / 60.0;
      let y = integration.solution_at(t).unwrap();
      assert!((y[0] - (-t).exp()).abs() < 1.0e-8, "frame {frame}: {y:?}");
    }
    assert!(integration.statistics().accepted < 200);
  }
}
