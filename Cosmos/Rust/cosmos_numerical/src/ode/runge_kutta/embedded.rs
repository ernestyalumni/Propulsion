//! The adaptive driver for explicit embedded Runge–Kutta pairs: HNW I §II.4
//! step-size selection, §II.6 dense output, and the stiffness test of the
//! DOPRI5/DOP853 codes, with the pair itself behind the `EmbeddedPair` trait.
//!
//! # One step
//!
//! From `(t, y)` with `f_0 = f(t, y)` known, the pair returns a proposal
//! `y_1` and a scaled error `err` (HNW I eq. (4.11)). With
//! `α = ERROR_EXPONENT − BETA_CORRECTION · β`, the next step is
//!
//! `h_new = h · clamp(safety · err^{−α} · err_previous^{β}, min_scale, max_scale)`,
//!
//! eq. (4.13) with Gustafsson's integral term (HNW II §IV.2), computed by
//! `PIStepSizeControl`. `err ≤ 1` accepts `y_1` (local extrapolation: the
//! higher-order result is kept) and remembers `err_previous = max(err, 10⁻⁴)`;
//! `err > 1` rejects, and the step after a rejection may not grow.
//!
//! The driver clamps the step so it lands on the end time, stops when the
//! step falls below `10 · u · |t|` (`u` the unit round-off), and treats a
//! non-finite error estimate as a rejection by the smallest allowed factor.
//!
//! # Stiffness
//!
//! After an accepted step the pair estimates `h|λ|`, `λ` the dominant
//! eigenvalue of `∂f/∂y`, from two stages at the same abscissa:
//! `h|λ| ≈ |h| ‖f(Y_a) − f(Y_b)‖ / ‖Y_a − Y_b‖`. Fifteen consecutive values
//! beyond the pair's stability boundary on the negative axis mark the problem
//! as stiff; six below it reset the count (Hairer's codes, HNW II §IV.2).
//!
//! # Dense output
//!
//! An accepted step can build an interpolant `u(θ)`, `θ = (t − t_start)/h`,
//! so solutions at arbitrary times (an animation frame, an output grid) cost
//! no extra steps (HNW I §II.6).

use core::fmt;

use super::initial_step::{initial_step_size, with_sign_of};
use super::pi_step_size::{PIStepSizeControl, PIStepSizeControlError};
use super::tolerance::Tolerance;
use crate::field::RealField;

/// The smallest `err_previous` remembered after an accepted step.
pub const MINIMUM_REMEMBERED_ERROR: f64 = 1.0e-4;
/// The integration stops if `STEP_UNDERFLOW_FACTOR · |h| ≤ u · |t|`.
pub const STEP_UNDERFLOW_FACTOR: f64 = 0.1;
/// A step within this fraction of the remaining interval is stretched to land
/// on the end time instead of leaving a sliver.
pub const LANDING_STRETCH: f64 = 1.01;
pub const DEFAULT_MAXIMUM_ATTEMPTS: usize = 100_000;
pub const DEFAULT_STIFFNESS_CHECK_INTERVAL: usize = 1000;
pub const STIFF_STEPS_TO_DECLARE_STIFFNESS: usize = 15;
pub const NONSTIFF_STEPS_TO_RESET: usize = 6;

/// A solution value at the scaled time `θ ∈ [0, 1]` of one accepted step.
pub trait Interpolant<T: RealField, const N: usize>
{
  fn evaluate(&self, theta: T) -> [T; N];
}

/// The result of one attempted step.
#[derive(Clone, Copy, Debug)]
pub struct Trial<T: RealField, const N: usize>
{
  pub y_new: [T; N],
  /// The scaled error, `err` of HNW I eq. (4.11) or the pair's own estimator.
  pub error: T,
}

/// An explicit embedded Runge–Kutta pair: its stages, error estimator,
/// stiffness estimate, interpolant and default step-control constants.
pub trait EmbeddedPair<T: RealField, const N: usize>
{
  const NAME: &'static str;
  /// `k` in the starting-step rule `h_1^k · max(d_1, d_2) = 0.01`.
  const STARTING_STEP_ORDER: u32;
  /// `1/(q+1)` of HNW I eq. (4.12), before the integral correction.
  const ERROR_EXPONENT: f64;
  /// `α = ERROR_EXPONENT − BETA_CORRECTION · β` (Hairer's `EXPO1`).
  const BETA_CORRECTION: f64;
  const DEFAULT_BETA: f64;
  const DEFAULT_MIN_SCALE: f64;
  const DEFAULT_MAX_SCALE: f64;
  const DEFAULT_SAFETY_FACTOR: f64 = 0.9;
  /// Where the stability region meets the negative real axis, about.
  const STIFFNESS_LIMIT: f64;
  const EVALUATIONS_PER_ATTEMPT: usize;
  const EVALUATIONS_AT_ACCEPTANCE: usize;
  const EVALUATIONS_FOR_INTERPOLANT: usize;

  type Interpolant: Interpolant<T, N> + Clone;

  /// Stages, proposal and error for the step `(t, y) → t + h`.
  fn attempt<F>(
    &mut self,
    rhs: &mut F,
    t: T,
    y: &[T; N],
    f_0: &[T; N],
    h: T,
    tolerance: &Tolerance<T, N>) -> Trial<T, N>
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized;

  /// `f(t + h, y_new)` for the attempt just accepted, which becomes the first
  /// stage of the next step.
  fn derivative_at_end<F>(
    &mut self,
    rhs: &mut F,
    t: T,
    h: T,
    y_new: &[T; N]) -> [T; N]
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized;

  /// `h|λ|` from the accepted step, `None` if the two stages coincide.
  fn stiffness_estimate(&self, h: T, y_new: &[T; N]) -> Option<T>;

  /// The dense-output polynomial of the accepted step.
  fn interpolant<F>(
    &mut self,
    rhs: &mut F,
    t: T,
    y: &[T; N],
    h: T,
    y_new: &[T; N]) -> Self::Interpolant
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized;

  /// The PI controller of the reference code, with integral gain `beta`.
  fn step_control_with_beta(
    beta: T) -> Result<PIStepSizeControl<T>, PIStepSizeControlError>
  {
    PIStepSizeControl::new(
      T::from_f64(Self::ERROR_EXPONENT)
        - beta * T::from_f64(Self::BETA_CORRECTION),
      beta,
      T::from_f64(Self::DEFAULT_MIN_SCALE),
      T::from_f64(Self::DEFAULT_MAX_SCALE),
      T::from_f64(Self::DEFAULT_SAFETY_FACTOR))
  }

  fn default_step_control() -> PIStepSizeControl<T>
  {
    Self::step_control_with_beta(T::from_f64(Self::DEFAULT_BETA))
      .expect("the default step-control constants are in their domain")
  }
}

/// What to do once the stiffness test fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StiffnessPolicy
{
  /// Record the time in `Statistics::stiffness_detected_at` and continue.
  Report,
  /// Return `IntegrationError::Stiff`, leaving the state at the step's start.
  Abort,
}

#[derive(Clone, Copy, Debug)]
pub struct IntegratorOptions<T: RealField>
{
  /// The first trial step; `None` uses the starting-step rule.
  pub initial_step: Option<T>,
  /// The largest step magnitude; `None` means the whole interval.
  pub maximum_step: Option<T>,
  pub maximum_attempts: usize,
  /// Test every this many accepted steps (and after each positive test);
  /// `None` disables the test.
  pub stiffness_check_interval: Option<usize>,
  pub stiffness_policy: StiffnessPolicy,
  /// Build an interpolant for every accepted step.
  pub dense_output: bool,
  /// `None` uses the pair's reference controller.
  pub step_control: Option<PIStepSizeControl<T>>,
  pub unit_roundoff: T,
}

impl<T: RealField> Default for IntegratorOptions<T>
{
  fn default() -> Self
  {
    Self {
      initial_step: None,
      maximum_step: None,
      maximum_attempts: DEFAULT_MAXIMUM_ATTEMPTS,
      stiffness_check_interval: Some(DEFAULT_STIFFNESS_CHECK_INTERVAL),
      stiffness_policy: StiffnessPolicy::Report,
      dense_output: true,
      step_control: None,
      unit_roundoff: T::machine_epsilon(),
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntegrationError<T: RealField>
{
  NonFiniteInitialCondition,
  NonFiniteInitialDerivative,
  NonPositiveInitialStep,
  NonPositiveMaximumStep,
  ZeroMaximumAttempts,
  ZeroStiffnessCheckInterval,
  UnitRoundoffOutOfRange,
  EndTimeBehindCurrentTime { time: T, end_time: T },
  StepSizeTooSmall { time: T, step_size: T },
  TooManyAttempts { time: T, attempts: usize },
  Stiff { time: T },
  NonFiniteDerivative { time: T },
  DenseOutputDisabled,
  NoAcceptedStep,
  OutsideLastStep { time: T, step_start: T, step_end: T },
}

impl<T: RealField> fmt::Display for IntegrationError<T>
{
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
  {
    write!(formatter, "{self:?}")
  }
}

impl<T: RealField> std::error::Error for IntegrationError<T> {}

/// Work counters. `rejected` counts every rejection, including those before
/// the first accepted step, which Hairer's `NREJCT` leaves out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Statistics<T: RealField>
{
  pub function_evaluations: usize,
  pub attempts: usize,
  pub accepted: usize,
  pub rejected: usize,
  pub stiffness_detected_at: Option<T>,
}

impl<T: RealField> Default for Statistics<T>
{
  fn default() -> Self
  {
    Self {
      function_evaluations: 0,
      attempts: 0,
      accepted: 0,
      rejected: 0,
      stiffness_detected_at: None,
    }
  }
}

/// One accepted step `[start_time, end_time]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptedStep<T: RealField>
{
  pub start_time: T,
  pub end_time: T,
  pub step_size: T,
  pub error: T,
}

#[derive(Clone, Debug)]
struct LastStep<T: RealField, I>
{
  start_time: T,
  step_size: T,
  interpolant: Option<I>,
}

#[derive(Clone, Copy, Debug)]
struct StiffnessMonitor<T: RealField>
{
  consecutive_stiff: usize,
  consecutive_nonstiff: usize,
  last_estimate: Option<T>,
}

impl<T: RealField> Default for StiffnessMonitor<T>
{
  fn default() -> Self
  {
    Self { consecutive_stiff: 0, consecutive_nonstiff: 0, last_estimate: None }
  }
}

/// An initial value problem `y' = f(t, y)`, `y(t_0) = y_0` on `[t_0, t_end]`
/// (or `[t_end, t_0]`), advanced one accepted step at a time.
pub struct EmbeddedRungeKutta<T, const N: usize, M, F>
where
  T: RealField,
  M: EmbeddedPair<T, N>,
  F: FnMut(T, &[T; N]) -> [T; N],
{
  method: M,
  rhs: F,
  tolerance: Tolerance<T, N>,
  controller: PIStepSizeControl<T>,
  options: IntegratorOptions<T>,
  maximum_step: T,
  time: T,
  state: [T; N],
  derivative: [T; N],
  end_time: T,
  direction: T,
  step_size: T,
  previous_error: T,
  was_rejected: bool,
  finished: bool,
  last_step: Option<LastStep<T, M::Interpolant>>,
  stiffness: StiffnessMonitor<T>,
  statistics: Statistics<T>,
}

fn all_finite<T: RealField, const N: usize>(values: &[T; N]) -> bool
{
  values.iter().all(|value| value.is_finite())
}

impl<T, const N: usize, M, F> EmbeddedRungeKutta<T, N, M, F>
where
  T: RealField,
  M: EmbeddedPair<T, N>,
  F: FnMut(T, &[T; N]) -> [T; N],
{
  /// Validates the problem, evaluates `f(t_0, y_0)` and, unless an initial
  /// step is given, the starting-step rule. An empty interval evaluates
  /// nothing and is finished at once.
  pub fn new(
    method: M,
    rhs: F,
    t_0: T,
    y_0: [T; N],
    end_time: T,
    tolerance: Tolerance<T, N>,
    options: IntegratorOptions<T>) -> Result<Self, IntegrationError<T>>
  {
    if !t_0.is_finite() || !end_time.is_finite() || !all_finite(&y_0)
    {
      return Err(IntegrationError::NonFiniteInitialCondition);
    }
    if options.maximum_attempts == 0
    {
      return Err(IntegrationError::ZeroMaximumAttempts);
    }
    if options.stiffness_check_interval == Some(0)
    {
      return Err(IntegrationError::ZeroStiffnessCheckInterval);
    }
    // Written so that NaN fails the test.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let unit_roundoff_out_of_range = !(options.unit_roundoff > T::from_f64(1.0e-35)
      && options.unit_roundoff < T::one());
    if unit_roundoff_out_of_range
    {
      return Err(IntegrationError::UnitRoundoffOutOfRange);
    }
    if let Some(initial_step) = options.initial_step
    {
      if !(initial_step.is_finite() && initial_step > T::zero())
      {
        return Err(IntegrationError::NonPositiveInitialStep);
      }
    }
    let interval = (end_time - t_0).absolute_value();
    let maximum_step = match options.maximum_step
    {
      Some(maximum_step) =>
      {
        if !(maximum_step.is_finite() && maximum_step > T::zero())
        {
          return Err(IntegrationError::NonPositiveMaximumStep);
        }
        maximum_step
      }
      None => interval,
    };
    let direction = if end_time < t_0 { -T::one() } else { T::one() };
    let controller = options.step_control.unwrap_or_else(M::default_step_control);

    let mut integrator = Self {
      method,
      rhs,
      tolerance,
      controller,
      options,
      maximum_step,
      time: t_0,
      state: y_0,
      derivative: [T::zero(); N],
      end_time,
      direction,
      step_size: T::zero(),
      previous_error: T::from_f64(MINIMUM_REMEMBERED_ERROR),
      was_rejected: false,
      finished: end_time == t_0,
      last_step: None,
      stiffness: StiffnessMonitor::default(),
      statistics: Statistics::default(),
    };
    if integrator.finished
    {
      return Ok(integrator);
    }

    integrator.initialize()?;
    Ok(integrator)
  }

  /// Evaluates `f` at the current point and chooses the first step: the
  /// given initial step, or the starting-step rule.
  fn initialize(&mut self) -> Result<(), IntegrationError<T>>
  {
    self.derivative = (self.rhs)(self.time, &self.state);
    self.statistics.function_evaluations += 1;
    if !all_finite(&self.derivative)
    {
      return Err(IntegrationError::NonFiniteInitialDerivative);
    }
    self.step_size = match self.options.initial_step
    {
      Some(initial_step) => with_sign_of(initial_step, self.direction),
      None =>
      {
        self.statistics.function_evaluations += 1;
        initial_step_size(
          &mut self.rhs,
          self.time,
          &self.state,
          &self.derivative,
          self.direction,
          M::STARTING_STEP_ORDER,
          self.maximum_step,
          &self.tolerance)
      }
    };
    if !self.step_size.is_finite() || self.step_size == T::zero()
    {
      return Err(IntegrationError::NonFiniteInitialDerivative);
    }
    Ok(())
  }

  pub fn time(&self) -> T { self.time }
  pub fn state(&self) -> &[T; N] { &self.state }
  /// `f(t, y)` at the current point (the first stage of the next step).
  pub fn derivative(&self) -> &[T; N] { &self.derivative }
  pub fn end_time(&self) -> T { self.end_time }
  /// The step size the next attempt will use.
  pub fn step_size(&self) -> T { self.step_size }
  pub fn is_finished(&self) -> bool { self.finished }
  pub fn statistics(&self) -> &Statistics<T> { &self.statistics }
  pub fn tolerance(&self) -> &Tolerance<T, N> { &self.tolerance }
  pub fn rhs(&self) -> &F { &self.rhs }

  /// Mutable access to the right-hand side, for parameters a user changes
  /// during the integration. Call `restart_from` after a change that makes
  /// `f` discontinuous at the current time.
  pub fn rhs_mut(&mut self) -> &mut F { &mut self.rhs }

  /// Restarts at the current time from `state` (or the current state):
  /// re-evaluates `f`, forgets the controller history and the last
  /// interpolant, keeps the step size. This is Hairer's `IRTRN = 2`, the
  /// right response to a discontinuity in `f` or a jump in `y`.
  pub fn restart_from(
    &mut self,
    state: Option<[T; N]>) -> Result<(), IntegrationError<T>>
  {
    if let Some(state) = state
    {
      if !all_finite(&state)
      {
        return Err(IntegrationError::NonFiniteInitialCondition);
      }
      self.state = state;
    }
    self.derivative = (self.rhs)(self.time, &self.state);
    self.statistics.function_evaluations += 1;
    if !all_finite(&self.derivative)
    {
      return Err(IntegrationError::NonFiniteDerivative { time: self.time });
    }
    self.previous_error = T::from_f64(MINIMUM_REMEMBERED_ERROR);
    self.was_rejected = false;
    self.last_step = None;
    self.stiffness = StiffnessMonitor::default();
    Ok(())
  }

  /// Moves the end time further in the direction of integration, so an
  /// open-ended (real-time) integration can keep going.
  pub fn extend_end_time(
    &mut self,
    end_time: T) -> Result<(), IntegrationError<T>>
  {
    if !end_time.is_finite()
      || (end_time - self.time) * self.direction < T::zero()
    {
      return Err(IntegrationError::EndTimeBehindCurrentTime {
        time: self.time,
        end_time,
      });
    }
    if self.options.maximum_step.is_none()
    {
      self.maximum_step = self
        .maximum_step
        .maximum((end_time - self.time).absolute_value());
    }
    self.end_time = end_time;
    self.finished = end_time == self.time;
    if self.step_size == T::zero() && !self.finished
    {
      self.initialize()?;
    }
    Ok(())
  }

  /// Takes one accepted step (rejecting and retrying as needed). `Ok(None)`
  /// once the end time has been reached.
  pub fn step(&mut self) -> Result<Option<AcceptedStep<T>>, IntegrationError<T>>
  {
    if self.finished
    {
      return Ok(None);
    }
    loop
    {
      if self.statistics.attempts >= self.options.maximum_attempts
      {
        return Err(IntegrationError::TooManyAttempts {
          time: self.time,
          attempts: self.statistics.attempts,
        });
      }
      if T::from_f64(STEP_UNDERFLOW_FACTOR) * self.step_size.absolute_value()
        <= self.time.absolute_value() * self.options.unit_roundoff
      {
        return Err(IntegrationError::StepSizeTooSmall {
          time: self.time,
          step_size: self.step_size,
        });
      }
      let mut h = self.step_size;
      let lands_on_end = (self.time + T::from_f64(LANDING_STRETCH) * h
        - self.end_time)
        * self.direction
        > T::zero();
      if lands_on_end
      {
        h = self.end_time - self.time;
      }
      self.statistics.attempts += 1;

      let trial = self.method.attempt(
        &mut self.rhs,
        self.time,
        &self.state,
        &self.derivative,
        h,
        &self.tolerance);
      self.statistics.function_evaluations += M::EVALUATIONS_PER_ATTEMPT;

      let acceptable = trial.error.is_finite() && all_finite(&trial.y_new);
      if acceptable && trial.error <= T::one()
      {
        return self.accept(trial, h, lands_on_end).map(Some);
      }

      self.step_size = if acceptable
      {
        self.controller.compute_new_step_size(
          trial.error, self.previous_error, h, self.was_rejected)
      }
      else
      {
        h * self.controller.min_scale()
      };
      self.was_rejected = true;
      self.statistics.rejected += 1;
    }
  }

  fn accept(
    &mut self,
    trial: Trial<T, N>,
    h: T,
    lands_on_end: bool) -> Result<AcceptedStep<T>, IntegrationError<T>>
  {
    let mut h_new = self.controller.compute_new_step_size(
      trial.error, self.previous_error, h, self.was_rejected);
    self.previous_error =
      trial.error.maximum(T::from_f64(MINIMUM_REMEMBERED_ERROR));
    self.statistics.accepted += 1;

    let derivative_at_end = self.method.derivative_at_end(
      &mut self.rhs, self.time, h, &trial.y_new);
    self.statistics.function_evaluations += M::EVALUATIONS_AT_ACCEPTANCE;
    if !all_finite(&derivative_at_end)
    {
      return Err(IntegrationError::NonFiniteDerivative { time: self.time + h });
    }

    self.test_stiffness(h, &trial.y_new)?;

    let interpolant = if self.options.dense_output
    {
      self.statistics.function_evaluations += M::EVALUATIONS_FOR_INTERPOLANT;
      Some(self.method.interpolant(
        &mut self.rhs, self.time, &self.state, h, &trial.y_new))
    }
    else
    {
      None
    };

    let start_time = self.time;
    self.time = if lands_on_end { self.end_time } else { self.time + h };
    self.state = trial.y_new;
    self.derivative = derivative_at_end;
    self.last_step = Some(LastStep { start_time, step_size: h, interpolant });

    if lands_on_end
    {
      self.finished = true;
    }
    else
    {
      if h_new.absolute_value() > self.maximum_step
      {
        h_new = with_sign_of(self.maximum_step, self.direction);
      }
    }
    self.step_size = h_new;
    self.was_rejected = false;

    Ok(AcceptedStep {
      start_time,
      end_time: self.time,
      step_size: h,
      error: trial.error,
    })
  }

  fn test_stiffness(&mut self, h: T, y_new: &[T; N])
    -> Result<(), IntegrationError<T>>
  {
    let Some(interval) = self.options.stiffness_check_interval
    else
    {
      return Ok(());
    };
    if !self.statistics.accepted.is_multiple_of(interval)
      && self.stiffness.consecutive_stiff == 0
    {
      return Ok(());
    }
    if let Some(estimate) = self.method.stiffness_estimate(h, y_new)
    {
      self.stiffness.last_estimate = Some(estimate);
    }
    let Some(estimate) = self.stiffness.last_estimate
    else
    {
      return Ok(());
    };
    if estimate > T::from_f64(M::STIFFNESS_LIMIT)
    {
      self.stiffness.consecutive_nonstiff = 0;
      self.stiffness.consecutive_stiff += 1;
      if self.stiffness.consecutive_stiff == STIFF_STEPS_TO_DECLARE_STIFFNESS
      {
        if self.statistics.stiffness_detected_at.is_none()
        {
          self.statistics.stiffness_detected_at = Some(self.time);
        }
        if self.options.stiffness_policy == StiffnessPolicy::Abort
        {
          return Err(IntegrationError::Stiff { time: self.time });
        }
      }
    }
    else
    {
      self.stiffness.consecutive_nonstiff += 1;
      if self.stiffness.consecutive_nonstiff == NONSTIFF_STEPS_TO_RESET
      {
        self.stiffness.consecutive_stiff = 0;
      }
    }
    Ok(())
  }

  /// The interval of the last accepted step, if any.
  pub fn last_step_interval(&self) -> Option<(T, T)>
  {
    self
      .last_step
      .as_ref()
      .map(|last| (last.start_time, last.start_time + last.step_size))
  }

  /// The dense output at `time` inside the last accepted step.
  pub fn dense_output(&self, time: T) -> Result<[T; N], IntegrationError<T>>
  {
    if !self.options.dense_output
    {
      return Err(IntegrationError::DenseOutputDisabled);
    }
    let last = self.last_step.as_ref().ok_or(IntegrationError::NoAcceptedStep)?;
    let interpolant =
      last.interpolant.as_ref().ok_or(IntegrationError::NoAcceptedStep)?;
    let theta = (time - last.start_time) / last.step_size;
    if !(theta >= T::zero() && theta <= T::one())
    {
      return Err(IntegrationError::OutsideLastStep {
        time,
        step_start: last.start_time,
        step_end: self.time,
      });
    }
    Ok(interpolant.evaluate(theta))
  }

  /// Steps until `time` is covered and returns the solution there: the
  /// current state if `time` is the current time, otherwise the dense output.
  /// This is the per-frame call of a real-time loop.
  pub fn solution_at(&mut self, time: T) -> Result<[T; N], IntegrationError<T>>
  {
    if (time - self.end_time) * self.direction > T::zero()
    {
      return Err(IntegrationError::EndTimeBehindCurrentTime {
        time,
        end_time: self.end_time,
      });
    }
    while (time - self.time) * self.direction > T::zero()
    {
      self.step()?;
    }
    if time == self.time
    {
      return Ok(self.state);
    }
    self.dense_output(time)
  }

  /// Steps to the end time.
  pub fn integrate_to_end(&mut self) -> Result<[T; N], IntegrationError<T>>
  {
    while self.step()?.is_some() {}
    Ok(self.state)
  }
}

/// `y + h Σ_j w_j k_j` over the first `count` stages, skipping zero weights,
/// summed in stage order (the reference codes' evaluation order).
pub(crate) fn stage_input<T: RealField, const N: usize, const S: usize>(
  y: &[T; N],
  h: T,
  weights: &[f64; S],
  stages: &[[T; N]; S],
  count: usize) -> [T; N]
{
  let mut input = [T::zero(); N];
  let mut nonzero = weights[..count].iter().enumerate().filter(|(_, w)| **w != 0.0);
  if let (Some((only, weight)), None) = (nonzero.next(), nonzero.next())
  {
    // A single-term row is `y + (h a) k` in the reference codes.
    let h_weight = h * T::from_f64(*weight);
    for i in 0..N
    {
      input[i] = y[i] + h_weight * stages[only][i];
    }
    return input;
  }
  let sum = weighted_sum(weights, stages, count);
  for i in 0..N
  {
    input[i] = y[i] + h * sum[i];
  }
  input
}

/// `Σ_j w_j k_j` over the first `count` stages, skipping zero weights.
pub(crate) fn weighted_sum<T: RealField, const N: usize, const S: usize>(
  weights: &[f64; S],
  stages: &[[T; N]; S],
  count: usize) -> [T; N]
{
  let mut sum = [T::zero(); N];
  for (weight, stage) in weights.iter().zip(stages.iter()).take(count)
  {
    if *weight == 0.0
    {
      continue;
    }
    let weight = T::from_f64(*weight);
    for i in 0..N
    {
      sum[i] = sum[i] + weight * stage[i];
    }
  }
  sum
}

/// `|h| ‖f_a − f_b‖ / ‖y_a − y_b‖`, the stiffness quotient.
pub(crate) fn stiffness_quotient<T: RealField, const N: usize>(
  h: T,
  f_a: &[T; N],
  f_b: &[T; N],
  y_a: &[T; N],
  y_b: &[T; N]) -> Option<T>
{
  let mut numerator = T::zero();
  let mut denominator = T::zero();
  for i in 0..N
  {
    numerator = numerator + (f_a[i] - f_b[i]) * (f_a[i] - f_b[i]);
    denominator = denominator + (y_a[i] - y_b[i]) * (y_a[i] - y_b[i]);
  }
  if denominator > T::zero()
  {
    Some(h.absolute_value() * (numerator / denominator).square_root())
  }
  else
  {
    None
  }
}
