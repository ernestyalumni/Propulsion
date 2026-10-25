//! A torque-free (or torqued) rigid body integrated four ways at once, plus a
//! DOP853 reference at a stringent tolerance, sampled at render times.
//!
//! # Real-time sampling
//!
//! The renderer asks for the state at wall-clock-driven simulation times
//! `t_frame`. The two fixed-step methods step with their own `Δt` and blend
//! the last two states (Fiedler's "fix your timestep": linear in `ω`,
//! spherical in `q`); the two adaptive methods take whatever steps their
//! tolerance needs and return their own dense output at `t_frame`, which is
//! an accurate solution value, not a blend.
//!
//! # Diagnostics
//!
//! Every body is compared against the reference run at the same time: the
//! relative error in kinetic energy and in `|L|`, and the attitude error
//! `2 arccos |⟨q, q_ref⟩|`. With no torque, `E` and `|L|` are invariants
//! (GNI §IV.1), so their errors are the integrators' drift; implicit
//! midpoint keeps both quadratic invariants (GNI §IV.2) and RK4 does not.

use std::cell::Cell;
use std::rc::Rc;

use cosmos_dynamics::rigid_body::algebra::{
  dot, normalized_quaternion, Quaternion, Vector3,
};
use cosmos_dynamics::rigid_body::{
  project_quaternion, ImplicitMidpoint, Model, State, Wrench,
  ANGULAR_VELOCITY_OFFSET, QUATERNION_OFFSET, STATE_DIMENSION,
};
use cosmos_numerical::ode::runge_kutta::classical_4::runge_kutta_4_step;
use cosmos_numerical::ode::runge_kutta::dormand_prince_5::DormandPrince5;
use cosmos_numerical::ode::runge_kutta::dormand_prince_853::DormandPrince853;
use cosmos_numerical::ode::runge_kutta::embedded::{
  EmbeddedPair, EmbeddedRungeKutta, IntegratorOptions,
};
use cosmos_numerical::ode::runge_kutta::tolerance::Tolerance;

type StateArray = [f64; STATE_DIMENSION];

/// Values per body in the frame buffer.
pub const FRAME_STRIDE: usize = 24;
pub const BODY_COUNT: usize = 4;
/// The four integrators, then the reference.
pub const RUN_COUNT: usize = BODY_COUNT + 1;
pub const REFERENCE_TOLERANCE: f64 = 1.0e-13;
/// The adaptive runs are open-ended; this horizon is extended as needed.
pub const HORIZON: f64 = 1.0e6;
/// A guard against a frame asking for more fixed steps than a frame can take.
pub const MAXIMUM_FIXED_STEPS_PER_FRAME: usize = 200_000;

/// Offsets inside one body's `FRAME_STRIDE` values.
pub mod frame
{
  pub const TIME: usize = 0;
  pub const QUATERNION: usize = 1;
  pub const ANGULAR_VELOCITY: usize = 5;
  pub const ANGULAR_MOMENTUM_INERTIAL: usize = 8;
  pub const KINETIC_ENERGY: usize = 11;
  pub const ENERGY_ERROR: usize = 12;
  pub const MOMENTUM_ERROR: usize = 13;
  pub const ATTITUDE_ERROR: usize = 14;
  pub const QUATERNION_NORM_ERROR: usize = 15;
  pub const STEP_SIZE: usize = 16;
  pub const ACCEPTED: usize = 17;
  pub const REJECTED: usize = 18;
  pub const EVALUATIONS: usize = 19;
  pub const FAILED: usize = 20;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method
{
  Classical4,
  ImplicitMidpoint,
  DormandPrince5,
  DormandPrince853,
}

pub const METHODS: [Method; BODY_COUNT] = [
  Method::Classical4,
  Method::ImplicitMidpoint,
  Method::DormandPrince5,
  Method::DormandPrince853,
];

impl Method
{
  pub fn name(self) -> &'static str
  {
    match self
    {
      Method::Classical4 => "RK4",
      Method::ImplicitMidpoint => "Implicit midpoint",
      Method::DormandPrince5 => "DOPRI5",
      Method::DormandPrince853 => "DOP853",
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario
{
  /// Spin about the intermediate principal axis: the Dzhanibekov flip.
  IntermediateAxis,
  /// An axisymmetric body: `ω` precesses about the symmetry axis.
  Precession,
  /// A general asymmetric tumble.
  Tumble,
}

impl Scenario
{
  pub fn from_index(index: u32) -> Option<Self>
  {
    match index
    {
      0 => Some(Scenario::IntermediateAxis),
      1 => Some(Scenario::Precession),
      2 => Some(Scenario::Tumble),
      _ => None,
    }
  }

  /// `(principal moments, initial body angular velocity)`.
  fn setup(self) -> (Vector3, Vector3)
  {
    match self
    {
      Scenario::IntermediateAxis => ([1.0, 2.0, 3.0], [5.0e-4, 4.0, 5.0e-4]),
      Scenario::Precession => ([2.0, 2.0, 3.0], [1.0, 0.0, 3.0]),
      Scenario::Tumble => ([1.0, 2.0, 3.0], [1.5, 0.3, 2.0]),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LabError
{
  InvalidModel(String),
  InvalidSetting(&'static str),
}

impl std::fmt::Display for LabError
{
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    write!(formatter, "{self:?}")
  }
}

/// What the lab needs from an adaptive integration, whatever its pair and
/// right-hand-side closure.
trait AdaptiveRun
{
  fn solution_at(&mut self, time: f64) -> Result<StateArray, String>;
  /// Restarts at `time` (inside the last step) after `f` changed there.
  fn restart_at(&mut self, time: f64) -> Result<(), String>;
  fn last_step_size(&self) -> f64;
  /// `(accepted, rejected, function evaluations)`.
  fn counts(&self) -> (usize, usize, usize);
}

impl<M, F> AdaptiveRun for EmbeddedRungeKutta<f64, STATE_DIMENSION, M, F>
where
  M: EmbeddedPair<f64, STATE_DIMENSION>,
  F: FnMut(f64, &StateArray) -> StateArray,
{
  fn solution_at(&mut self, time: f64) -> Result<StateArray, String>
  {
    if time > self.end_time()
    {
      self
        .extend_end_time(time + HORIZON)
        .map_err(|error| error.to_string())?;
    }
    EmbeddedRungeKutta::solution_at(self, time).map_err(|error| error.to_string())
  }

  fn restart_at(&mut self, time: f64) -> Result<(), String>
  {
    EmbeddedRungeKutta::restart_at(self, time).map_err(|error| error.to_string())
  }

  fn last_step_size(&self) -> f64
  {
    self
      .last_step_interval()
      .map_or(self.step_size(), |(start, end)| end - start)
  }

  fn counts(&self) -> (usize, usize, usize)
  {
    let statistics = self.statistics();
    (statistics.accepted, statistics.rejected, statistics.function_evaluations)
  }
}

// The fixed-step variant holds two states inline; one per body, so boxing
// it would buy nothing.
#[allow(clippy::large_enum_variant)]
enum Propagator
{
  Fixed
  {
    method: Method,
    previous: StateArray,
    current: StateArray,
    current_time: f64,
    steps: usize,
  },
  Adaptive(Box<dyn AdaptiveRun>),
}

struct Run
{
  propagator: Propagator,
  failed: bool,
}

pub struct RigidBodyLab
{
  model: Model,
  wrench: Rc<Cell<Wrench>>,
  time_step: f64,
  runs: Vec<Run>,
  midpoint: ImplicitMidpoint,
  frame: Vec<f64>,
  time: f64,
}

fn adaptive_run<M>(
  model: &Model,
  wrench: &Rc<Cell<Wrench>>,
  initial: StateArray,
  tolerance: f64) -> Result<Box<dyn AdaptiveRun>, LabError>
where
  M: EmbeddedPair<f64, STATE_DIMENSION> + Default + 'static,
{
  let model = model.clone();
  let wrench = Rc::clone(wrench);
  let rhs = move |_t: f64, y: &StateArray| model.derivative_array(y, &wrench.get());
  let options = IntegratorOptions {
    maximum_attempts: usize::MAX,
    ..IntegratorOptions::default()
  };
  let tolerance = Tolerance::uniform(tolerance, tolerance)
    .map_err(|_| LabError::InvalidSetting("tolerance must be positive"))?;
  let run = EmbeddedRungeKutta::new(
    M::default(), rhs, 0.0, initial, HORIZON, tolerance, options)
    .map_err(|error| LabError::InvalidModel(error.to_string()))?;
  Ok(Box::new(run))
}

/// `q(α)` on the great arc from `a` to `b` (shorter way), `α ∈ [0, 1]`.
fn slerp(a: &Quaternion, b: &Quaternion, alpha: f64) -> Quaternion
{
  let mut b = *b;
  let mut cosine = dot4(a, &b);
  if cosine < 0.0
  {
    b = b.map(|component| -component);
    cosine = -cosine;
  }
  let (weight_a, weight_b) = if cosine > 0.9995
  {
    (1.0 - alpha, alpha)
  }
  else
  {
    let angle = cosine.acos();
    let sine = angle.sin();
    (((1.0 - alpha) * angle).sin() / sine, (alpha * angle).sin() / sine)
  };
  let mut q = [0.0; 4];
  for i in 0..4
  {
    q[i] = weight_a * a[i] + weight_b * b[i];
  }
  normalized_quaternion(&q).unwrap_or(*a)
}

fn dot4(a: &Quaternion, b: &Quaternion) -> f64
{
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn quaternion_of(y: &StateArray) -> Quaternion
{
  [
    y[QUATERNION_OFFSET],
    y[QUATERNION_OFFSET + 1],
    y[QUATERNION_OFFSET + 2],
    y[QUATERNION_OFFSET + 3],
  ]
}

fn angular_velocity_of(y: &StateArray) -> Vector3
{
  [
    y[ANGULAR_VELOCITY_OFFSET],
    y[ANGULAR_VELOCITY_OFFSET + 1],
    y[ANGULAR_VELOCITY_OFFSET + 2],
  ]
}

impl RigidBodyLab
{
  pub fn new(
    scenario: Scenario,
    time_step: f64,
    tolerance: f64) -> Result<Self, LabError>
  {
    if !(time_step.is_finite() && time_step > 0.0)
    {
      return Err(LabError::InvalidSetting("time step must be positive"));
    }
    let (moments, angular_velocity) = scenario.setup();
    let inertia = [
      [moments[0], 0.0, 0.0],
      [0.0, moments[1], 0.0],
      [0.0, 0.0, moments[2]],
    ];
    let model = Model::new(1.0, inertia)
      .map_err(|error| LabError::InvalidModel(error.to_string()))?;
    let initial = State {
      angular_velocity_body: angular_velocity,
      ..State::default()
    }
    .to_array();
    let wrench = Rc::new(Cell::new(Wrench::default()));

    let mut runs = Vec::with_capacity(RUN_COUNT);
    for method in METHODS
    {
      let propagator = match method
      {
        Method::Classical4 | Method::ImplicitMidpoint => Propagator::Fixed {
          method,
          previous: initial,
          current: initial,
          current_time: 0.0,
          steps: 0,
        },
        Method::DormandPrince5 => Propagator::Adaptive(
          adaptive_run::<DormandPrince5<f64, STATE_DIMENSION>>(
            &model, &wrench, initial, tolerance)?),
        Method::DormandPrince853 => Propagator::Adaptive(
          adaptive_run::<DormandPrince853<f64, STATE_DIMENSION>>(
            &model, &wrench, initial, tolerance)?),
      };
      runs.push(Run { propagator, failed: false });
    }
    runs.push(Run {
      propagator: Propagator::Adaptive(
        adaptive_run::<DormandPrince853<f64, STATE_DIMENSION>>(
          &model, &wrench, initial, REFERENCE_TOLERANCE)?),
      failed: false,
    });

    let mut lab = Self {
      model,
      wrench,
      time_step,
      runs,
      midpoint: ImplicitMidpoint::default(),
      frame: vec![0.0; RUN_COUNT * FRAME_STRIDE],
      time: 0.0,
    };
    lab.advance_to(0.0);
    Ok(lab)
  }

  pub fn frame(&self) -> &[f64] { &self.frame }
  pub fn time_step(&self) -> f64 { self.time_step }

  /// A constant body-frame torque from the last frame time on. The adaptive
  /// runs restart there from their dense output, since `f` jumps (Hairer's
  /// `IRTRN = 2`); the fixed-step runs take it from their next step, so their
  /// switch time is late by less than one `Δt`.
  pub fn set_torque(&mut self, torque: Vector3)
  {
    if !torque.iter().all(|component| component.is_finite())
    {
      return;
    }
    self.wrench.set(Wrench { torque_body: torque, ..Wrench::default() });
    for run in &mut self.runs
    {
      if let Propagator::Adaptive(adaptive) = &mut run.propagator
      {
        if adaptive.restart_at(self.time).is_err()
        {
          run.failed = true;
        }
      }
    }
  }

  fn fixed_step(&self, method: Method, y: &StateArray) -> Option<StateArray>
  {
    let wrench = self.wrench.get();
    match method
    {
      Method::Classical4 =>
      {
        let rhs = |_t: f64, y: &StateArray| self.model.derivative_array(y, &wrench);
        let mut next = runge_kutta_4_step(&rhs, 0.0, y, self.time_step);
        project_quaternion(&mut next).ok()?;
        Some(next)
      }
      Method::ImplicitMidpoint => self
        .midpoint
        .step(&self.model, &State::from_array(y), self.time_step, &wrench)
        .ok()
        .map(|step| step.state.to_array()),
      _ => None,
    }
  }

  /// Advances every run to `time` and refreshes the frame buffer.
  pub fn advance_to(&mut self, time: f64)
  {
    self.time = time;
    let mut states = [[0.0; STATE_DIMENSION]; RUN_COUNT];
    let mut render_quaternions = [[1.0, 0.0, 0.0, 0.0]; RUN_COUNT];
    for index in 0..RUN_COUNT
    {
      let (state, render) = self.advance_run(index, time);
      states[index] = state;
      render_quaternions[index] = render;
    }

    let reference = states[BODY_COUNT];
    let reference_state = State::from_array(&reference);
    let reference_energy =
      self.model.kinetic_energy(&reference_state).unwrap_or(f64::NAN);
    let reference_momentum = self
      .model
      .angular_momentum_inertial(&reference_state)
      .map_or(f64::NAN, |momentum| dot(&momentum, &momentum).sqrt());
    let reference_quaternion = quaternion_of(&reference);

    for index in 0..RUN_COUNT
    {
      self.write_frame(
        index,
        time,
        &states[index],
        &render_quaternions[index],
        reference_energy,
        reference_momentum,
        &reference_quaternion);
    }
  }

  /// `(integrated state at time, quaternion to render)`.
  fn advance_run(&mut self, index: usize, time: f64) -> (StateArray, Quaternion)
  {
    if self.runs[index].failed
    {
      return ([f64::NAN; STATE_DIMENSION], [1.0, 0.0, 0.0, 0.0]);
    }
    let time_step = self.time_step;
    let outcome = match &self.runs[index].propagator
    {
      Propagator::Fixed { method, previous, current, current_time, steps } =>
      {
        let (method, mut previous, mut current, mut current_time, mut steps) =
          (*method, *previous, *current, *current_time, *steps);
        let mut taken = 0;
        let mut failed = false;
        while current_time < time && taken < MAXIMUM_FIXED_STEPS_PER_FRAME
        {
          match self.fixed_step(method, &current)
          {
            Some(next) =>
            {
              previous = current;
              current = next;
              current_time += time_step;
              steps += 1;
              taken += 1;
            }
            None =>
            {
              failed = true;
              break;
            }
          }
        }
        self.runs[index].propagator =
          Propagator::Fixed { method, previous, current, current_time, steps };
        if failed
        {
          None
        }
        else
        {
          let alpha = if steps == 0
          {
            1.0
          }
          else
          {
            (1.0 - (current_time - time) / time_step).clamp(0.0, 1.0)
          };
          let render =
            slerp(&quaternion_of(&previous), &quaternion_of(&current), alpha);
          Some((current, render))
        }
      }
      Propagator::Adaptive(_) => None,
    };
    if let Propagator::Adaptive(adaptive) = &mut self.runs[index].propagator
    {
      return match adaptive.solution_at(time)
      {
        Ok(state) =>
        {
          let render = normalized_quaternion(&quaternion_of(&state))
            .unwrap_or([1.0, 0.0, 0.0, 0.0]);
          (state, render)
        }
        Err(_) =>
        {
          self.runs[index].failed = true;
          ([f64::NAN; STATE_DIMENSION], [1.0, 0.0, 0.0, 0.0])
        }
      };
    }
    match outcome
    {
      Some(result) => result,
      None =>
      {
        self.runs[index].failed = true;
        ([f64::NAN; STATE_DIMENSION], [1.0, 0.0, 0.0, 0.0])
      }
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn write_frame(
    &mut self,
    index: usize,
    time: f64,
    state: &StateArray,
    render_quaternion: &Quaternion,
    reference_energy: f64,
    reference_momentum: f64,
    reference_quaternion: &Quaternion)
  {
    let base = index * FRAME_STRIDE;
    let run = &self.runs[index];
    let mut values = [0.0; FRAME_STRIDE];
    values[frame::TIME] = time;
    values[frame::FAILED] = if run.failed { 1.0 } else { 0.0 };

    let quaternion = quaternion_of(state);
    let norm = dot4(&quaternion, &quaternion).sqrt();
    values[frame::QUATERNION..frame::QUATERNION + 4]
      .copy_from_slice(render_quaternion);
    values[frame::ANGULAR_VELOCITY..frame::ANGULAR_VELOCITY + 3]
      .copy_from_slice(&angular_velocity_of(state));
    values[frame::QUATERNION_NORM_ERROR] = norm - 1.0;

    // Invariants of the integrated state with q normalized (the attitude
    // the integrator claims), so |q| drift is reported separately.
    let mut normalized = *state;
    if let Some(unit) = normalized_quaternion(&quaternion)
    {
      normalized[QUATERNION_OFFSET..QUATERNION_OFFSET + 4].copy_from_slice(&unit);
    }
    let body_state = State::from_array(&normalized);
    let energy = self.model.kinetic_energy(&body_state).unwrap_or(f64::NAN);
    let momentum = self
      .model
      .angular_momentum_inertial(&body_state)
      .unwrap_or([f64::NAN; 3]);
    values[frame::ANGULAR_MOMENTUM_INERTIAL..frame::ANGULAR_MOMENTUM_INERTIAL + 3]
      .copy_from_slice(&momentum);
    values[frame::KINETIC_ENERGY] = energy;
    values[frame::ENERGY_ERROR] = (energy - reference_energy) / reference_energy;
    values[frame::MOMENTUM_ERROR] =
      (dot(&momentum, &momentum).sqrt() - reference_momentum) / reference_momentum;
    let unit = normalized_quaternion(&quaternion).unwrap_or([f64::NAN; 4]);
    values[frame::ATTITUDE_ERROR] =
      2.0 * dot4(&unit, reference_quaternion).abs().min(1.0).acos();

    match &run.propagator
    {
      Propagator::Fixed { steps, .. } =>
      {
        values[frame::STEP_SIZE] = self.time_step;
        values[frame::ACCEPTED] = *steps as f64;
        let evaluations_per_step = match METHODS.get(index)
        {
          Some(Method::Classical4) => 4.0,
          _ => 0.0,
        };
        values[frame::EVALUATIONS] = evaluations_per_step * *steps as f64;
      }
      Propagator::Adaptive(adaptive) =>
      {
        let (accepted, rejected, evaluations) = adaptive.counts();
        values[frame::STEP_SIZE] = adaptive.last_step_size();
        values[frame::ACCEPTED] = accepted as f64;
        values[frame::REJECTED] = rejected as f64;
        values[frame::EVALUATIONS] = evaluations as f64;
      }
    }
    self.frame[base..base + FRAME_STRIDE].copy_from_slice(&values);
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  fn body(lab: &RigidBodyLab, index: usize, offset: usize) -> f64
  {
    lab.frame()[index * FRAME_STRIDE + offset]
  }

  /// Sixty frames a second for twenty seconds of the Dzhanibekov flip: the
  /// structure-preserving and adaptive runs stay on the invariants, RK4 with
  /// a coarse step drifts, and the flip happens.
  #[test]
  fn twenty_seconds_of_the_intermediate_axis_flip()
  {
    let mut lab = RigidBodyLab::new(Scenario::IntermediateAxis, 0.05, 1.0e-8)
      .unwrap();
    let mut smallest_spin = f64::INFINITY;
    for frame_number in 1..=1200
    {
      let time = frame_number as f64 / 60.0;
      lab.advance_to(time);
      smallest_spin = smallest_spin
        .min(body(&lab, BODY_COUNT, frame::ANGULAR_VELOCITY + 1));
    }
    assert!(smallest_spin < -3.9, "the flip: ω_2 reaches {smallest_spin}");
    for index in 0..RUN_COUNT
    {
      assert_eq!(body(&lab, index, frame::FAILED), 0.0, "run {index} failed");
    }
    let midpoint_energy = body(&lab, 1, frame::ENERGY_ERROR).abs();
    let classical_energy = body(&lab, 0, frame::ENERGY_ERROR).abs();
    let dop853_attitude = body(&lab, 3, frame::ATTITUDE_ERROR);
    assert!(midpoint_energy < 1.0e-11, "{midpoint_energy}");
    assert!(classical_energy > 1.0e3 * midpoint_energy, "{classical_energy}");
    assert!(dop853_attitude < 1.0e-5, "{dop853_attitude}");
    assert!(body(&lab, 3, frame::EVALUATIONS) < body(&lab, 2, frame::EVALUATIONS));
  }

  #[test]
  fn a_torque_restarts_the_adaptive_runs_and_they_follow_the_reference()
  {
    let mut lab =
      RigidBodyLab::new(Scenario::Tumble, 0.01, 1.0e-10).unwrap();
    lab.advance_to(1.0);
    lab.set_torque([0.0, 0.0, 0.5]);
    lab.advance_to(3.0);
    for index in [2, 3]
    {
      let attitude = body(&lab, index, frame::ATTITUDE_ERROR);
      assert!(attitude < 1.0e-6, "run {index}: {attitude}");
    }
    let energy = body(&lab, BODY_COUNT, frame::KINETIC_ENERGY);
    assert!(energy.is_finite());
  }

  #[test]
  fn the_blend_between_fixed_steps_is_a_unit_quaternion()
  {
    let a = [1.0, 0.0, 0.0, 0.0];
    let b = normalized_quaternion(&[0.8, 0.6, 0.0, 0.0]).unwrap();
    for alpha in [0.0, 0.3, 0.5, 1.0]
    {
      let q = slerp(&a, &b, alpha);
      assert!((dot4(&q, &q) - 1.0).abs() < 1.0e-15);
    }
    assert_eq!(slerp(&a, &b, 1.0), b);
  }
}
