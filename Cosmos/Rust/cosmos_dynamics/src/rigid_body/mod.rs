//! Six-degree-of-freedom rigid-body dynamics about the centre of mass, f64.
//!
//! # Conventions
//!
//! Position `r` and velocity `v` are inertial components. The attitude is a
//! scalar-first Hamilton unit quaternion `q` that maps body components to
//! inertial components, `x_I = R(q) x_B`. The angular velocity `ω` and the
//! inertia tensor `J` (about the centre of mass) are body components.
//!
//! # Equations of motion
//!
//! ```text
//!   r' = v
//!   v' = (R(q) F_B + F_I) / m
//!   q' = ½ q ⊗ (0, ω)
//!   J ω' = τ_B − ω × J ω
//! ```
//!
//! with `F_B` a body-fixed force, `F_I` an inertial force and `τ_B` the body
//! torque about the centre of mass (a [`Wrench`]). Nothing else is implicit:
//! no gravity, no damping.
//!
//! # State layout for the integrators
//!
//! [`State::to_array`] packs `[r (0..3), v (3..6), q = (w, x, y, z) (6..10),
//! ω (10..13)]` into `[f64; STATE_DIMENSION]`, and
//! [`Model::derivative_array`] is the right-hand side on that array, so a
//! general integrator is driven by
//! `|_t, y| model.derivative_array(y, &wrench)`. A general Runge–Kutta step
//! does not keep `|q| = 1`; [`project_quaternion`] puts it back after each
//! accepted step (HNW, *Geometric Numerical Integration*, §IV.4).
//!
//! The torque-free flow conserves the kinetic energy and the inertial
//! angular momentum, both quadratic in `J ω`. [`ImplicitMidpoint`] conserves
//! both exactly (up to its Newton tolerance and round-off); explicit
//! Runge–Kutta methods conserve neither (GNI §IV.2).
//!
//! Semantics follow `mathphysics/Rust/lie-groups/src/rigid_body.rs`; the
//! golden test reproduces its fixtures.

pub mod algebra;
mod implicit_midpoint;

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;

use algebra::{
  add, cross, determinant, dot, is_finite_vector, matrix_product,
  matrix_vector_product, normalized_quaternion, quaternion_norm_squared,
  rotation_matrix, subtract, trace, Matrix3, Quaternion, Vector3,
  IDENTITY_MATRIX, IDENTITY_QUATERNION,
};
use cosmos_numerical::ode::runge_kutta::classical_4::runge_kutta_4_step;

pub use implicit_midpoint::{ImplicitMidpoint, ImplicitMidpointStep};

/// `[r, v, q, ω]`: 3 + 3 + 4 + 3.
pub const STATE_DIMENSION: usize = 13;

pub const POSITION_OFFSET: usize = 0;
pub const VELOCITY_OFFSET: usize = 3;
pub const QUATERNION_OFFSET: usize = 6;
pub const ANGULAR_VELOCITY_OFFSET: usize = 10;

/// A value outside its physical or mathematical domain, or a numerical
/// failure that would otherwise leak a meaningless state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RigidBodyError
{
  NonPositiveMass,
  NonFiniteInput,
  ZeroInertia,
  InertiaNotSymmetric,
  InertiaNotPositiveDefinite,
  InertiaViolatesTriangleInequality,
  QuaternionNotUnit,
  NonPositiveTimeStep,
  InvalidNewtonTolerance,
  InvalidNewtonIterationLimit,
  NewtonDidNotConverge,
  SingularNewtonJacobian,
  NonFiniteResult,
}

impl core::fmt::Display for RigidBodyError
{
  fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
  {
    write!(formatter, "{self:?}")
  }
}

impl std::error::Error for RigidBodyError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State
{
  pub position: Vector3,
  pub velocity: Vector3,
  pub quaternion: Quaternion,
  pub angular_velocity_body: Vector3,
}

impl Default for State
{
  fn default() -> Self
  {
    Self
    {
      position: [0.0; 3],
      velocity: [0.0; 3],
      quaternion: IDENTITY_QUATERNION,
      angular_velocity_body: [0.0; 3],
    }
  }
}

impl State
{
  pub fn to_array(&self) -> [f64; STATE_DIMENSION]
  {
    let mut y = [0.0; STATE_DIMENSION];
    y[POSITION_OFFSET..VELOCITY_OFFSET].copy_from_slice(&self.position);
    y[VELOCITY_OFFSET..QUATERNION_OFFSET].copy_from_slice(&self.velocity);
    y[QUATERNION_OFFSET..ANGULAR_VELOCITY_OFFSET]
      .copy_from_slice(&self.quaternion);
    y[ANGULAR_VELOCITY_OFFSET..STATE_DIMENSION]
      .copy_from_slice(&self.angular_velocity_body);
    y
  }

  pub fn from_array(y: &[f64; STATE_DIMENSION]) -> Self
  {
    Self
    {
      position: [y[0], y[1], y[2]],
      velocity: [y[3], y[4], y[5]],
      quaternion: [y[6], y[7], y[8], y[9]],
      angular_velocity_body: [y[10], y[11], y[12]],
    }
  }
}

/// The time derivative of each [`State`] component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateDerivative
{
  pub position: Vector3,
  pub velocity: Vector3,
  pub quaternion: Quaternion,
  pub angular_velocity_body: Vector3,
}

impl StateDerivative
{
  pub fn to_array(&self) -> [f64; STATE_DIMENSION]
  {
    State
    {
      position: self.position,
      velocity: self.velocity,
      quaternion: self.quaternion,
      angular_velocity_body: self.angular_velocity_body,
    }
    .to_array()
  }
}

/// Forces and torque about the centre of mass, held fixed over a step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wrench
{
  pub force_body: Vector3,
  pub torque_body: Vector3,
  pub force_inertial: Vector3,
}

impl Wrench
{
  fn validate(&self) -> Result<(), RigidBodyError>
  {
    if is_finite_vector(&self.force_body)
      && is_finite_vector(&self.torque_body)
      && is_finite_vector(&self.force_inertial)
    {
      Ok(())
    }
    else
    {
      Err(RigidBodyError::NonFiniteInput)
    }
  }
}

/// A state with finite components and `| |q|² − 1 | <=`
/// [`Model::UNIT_QUATERNION_TOLERANCE`], returned with `q` renormalized.
fn validated(state: &State) -> Result<State, RigidBodyError>
{
  if !is_finite_vector(&state.position)
    || !is_finite_vector(&state.velocity)
    || !is_finite_vector(&state.angular_velocity_body)
    || state.quaternion.iter().any(|value| !value.is_finite())
  {
    return Err(RigidBodyError::NonFiniteInput);
  }
  let squared = quaternion_norm_squared(&state.quaternion);
  if !squared.is_finite()
    || (squared - 1.0).abs() > Model::UNIT_QUATERNION_TOLERANCE
  {
    return Err(RigidBodyError::QuaternionNotUnit);
  }
  let mut result = *state;
  result.quaternion = normalized_quaternion(&state.quaternion)
    .ok_or(RigidBodyError::QuaternionNotUnit)?;
  Ok(result)
}

fn finite_or_error(v: Vector3) -> Result<Vector3, RigidBodyError>
{
  if is_finite_vector(&v) { Ok(v) } else { Err(RigidBodyError::NonFiniteResult) }
}

/// The largest eigenvalue of a symmetric 3x3 matrix, by the trigonometric
/// solution of its characteristic cubic.
fn largest_symmetric_eigenvalue(a: &Matrix3) -> f64
{
  let off_diagonal = a[0][1].powi(2) + a[0][2].powi(2) + a[1][2].powi(2);
  if off_diagonal == 0.0
  {
    return a[0][0].max(a[1][1]).max(a[2][2]);
  }
  let mean = trace(a) / 3.0;
  let mut shifted: Matrix3 = core::array::from_fn(|i| {
    core::array::from_fn(|j| a[i][j] - IDENTITY_MATRIX[i][j] * mean)
  });
  let spread = ((shifted[0][0].powi(2)
    + shifted[1][1].powi(2)
    + shifted[2][2].powi(2)
    + 2.0 * off_diagonal)
    / 6.0)
    .sqrt();
  if spread == 0.0
  {
    return mean;
  }
  for row in shifted.iter_mut()
  {
    for value in row.iter_mut()
    {
      *value *= 1.0 / spread;
    }
  }
  let angle = (determinant(&shifted) / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
  mean + 2.0 * spread * angle.cos()
}

/// Mass and inertia of one rigid body.
#[derive(Clone, Debug)]
pub struct Model
{
  mass: f64,
  inertia: Matrix3,
  /// Largest |J_ij|; the Cholesky factor is of `J / inertia_scale`.
  inertia_scale: f64,
  cholesky_factor: Matrix3,
  inverse_inertia: Matrix3,
}

impl Model
{
  /// Relative asymmetry `|J_ij − J_ji| / max|J|` accepted (and averaged out).
  pub const SYMMETRY_TOLERANCE: f64 = 1.0e-12;
  /// Slack in the triangle inequality `2 λ_max <= tr J` (scaled).
  pub const TRIANGLE_TOLERANCE: f64 = 1.0e-12;
  /// Accepted `| |q|² − 1 |` for a state handed to a fallible method.
  pub const UNIT_QUATERNION_TOLERANCE: f64 = 1.0e-10;

  /// `inertia` is the full inertia tensor about the centre of mass in body
  /// axes. It must be symmetric, positive definite and satisfy the triangle
  /// inequality of principal moments, `J_a + J_b >= J_c`, which every mass
  /// distribution obeys (equivalently `2 λ_max <= tr J`).
  pub fn new(mass: f64, inertia: Matrix3) -> Result<Self, RigidBodyError>
  {
    if !mass.is_finite()
    {
      return Err(RigidBodyError::NonFiniteInput);
    }
    if mass <= 0.0
    {
      return Err(RigidBodyError::NonPositiveMass);
    }
    let mut inertia = inertia;
    let mut inertia_scale: f64 = 0.0;
    for value in inertia.iter().flatten()
    {
      if !value.is_finite()
      {
        return Err(RigidBodyError::NonFiniteInput);
      }
      inertia_scale = inertia_scale.max(value.abs());
    }
    if inertia_scale == 0.0
    {
      return Err(RigidBodyError::ZeroInertia);
    }
    for i in 0..3
    {
      for j in i + 1..3
      {
        if (inertia[i][j] / inertia_scale - inertia[j][i] / inertia_scale).abs()
          > Self::SYMMETRY_TOLERANCE
        {
          return Err(RigidBodyError::InertiaNotSymmetric);
        }
        let mean = inertia[i][j] / 2.0 + inertia[j][i] / 2.0;
        inertia[i][j] = mean;
        inertia[j][i] = mean;
      }
    }
    let scaled: Matrix3 = core::array::from_fn(|i| {
      core::array::from_fn(|j| inertia[i][j] / inertia_scale)
    });
    let mut cholesky_factor = [[0.0; 3]; 3];
    for i in 0..3
    {
      for j in 0..=i
      {
        let mut value = scaled[i][j];
        for k in 0..j
        {
          value -= cholesky_factor[i][k] * cholesky_factor[j][k];
        }
        if i == j
        {
          if !value.is_finite() || value <= 0.0
          {
            return Err(RigidBodyError::InertiaNotPositiveDefinite);
          }
          cholesky_factor[i][j] = value.sqrt();
        }
        else
        {
          cholesky_factor[i][j] = value / cholesky_factor[j][j];
        }
      }
    }
    if 2.0 * largest_symmetric_eigenvalue(&scaled)
      > trace(&scaled) + Self::TRIANGLE_TOLERANCE
    {
      return Err(RigidBodyError::InertiaViolatesTriangleInequality);
    }
    let mut model = Self
    {
      mass,
      inertia,
      inertia_scale,
      cholesky_factor,
      inverse_inertia: [[0.0; 3]; 3],
    };
    for j in 0..3
    {
      let mut unit = [0.0; 3];
      unit[j] = 1.0;
      let column = model.solve_inertia(&unit)?;
      for (i, value) in column.into_iter().enumerate()
      {
        model.inverse_inertia[i][j] = value;
      }
    }
    Ok(model)
  }

  pub fn mass(&self) -> f64 { self.mass }
  pub fn inertia(&self) -> &Matrix3 { &self.inertia }
  pub fn inverse_inertia(&self) -> &Matrix3 { &self.inverse_inertia }

  /// `J⁻¹ b` by forward and back substitution with the Cholesky factor.
  pub fn solve_inertia(&self, b: &Vector3) -> Result<Vector3, RigidBodyError>
  {
    if !is_finite_vector(b)
    {
      return Err(RigidBodyError::NonFiniteInput);
    }
    let factor = &self.cholesky_factor;
    let mut forward = [0.0; 3];
    let mut solution = [0.0; 3];
    for i in 0..3
    {
      let mut value = b[i] / self.inertia_scale;
      for (j, previous) in forward.iter().enumerate().take(i)
      {
        value -= factor[i][j] * previous;
      }
      forward[i] = value / factor[i][i];
    }
    for i in (0..3).rev()
    {
      let mut value = forward[i];
      for (j, later) in solution.iter().enumerate().skip(i + 1)
      {
        value -= factor[j][i] * later;
      }
      solution[i] = value / factor[i][i];
    }
    finite_or_error(solution)
  }

  /// The right-hand side on the packed state, for use inside an integrator.
  ///
  /// No validation and no renormalization: `R(q)` is evaluated from the
  /// quaternion as given, so drift off `|q| = 1` stays visible to the caller
  /// (and is removed by [`project_quaternion`]). Non-finite input propagates
  /// as NaN, which the integrator's own finiteness checks then report.
  pub fn derivative_array(
    &self,
    y: &[f64; STATE_DIMENSION],
    wrench: &Wrench) -> [f64; STATE_DIMENSION]
  {
    let velocity = [y[3], y[4], y[5]];
    let quaternion = [y[6], y[7], y[8], y[9]];
    let angular_velocity = [y[10], y[11], y[12]];

    let force = add(
      &matrix_vector_product(&rotation_matrix(&quaternion), &wrench.force_body),
      &wrench.force_inertial);
    let pure = [0.0, angular_velocity[0], angular_velocity[1], angular_velocity[2]];
    let quaternion_rate = algebra::hamilton_product(&quaternion, &pure);
    let angular_momentum = matrix_vector_product(&self.inertia, &angular_velocity);
    let angular_acceleration = matrix_vector_product(
      &self.inverse_inertia,
      &subtract(&wrench.torque_body, &cross(&angular_velocity, &angular_momentum)));

    let mut derivative = [0.0; STATE_DIMENSION];
    derivative[0..3].copy_from_slice(&velocity);
    for i in 0..3
    {
      derivative[VELOCITY_OFFSET + i] = force[i] / self.mass;
      derivative[ANGULAR_VELOCITY_OFFSET + i] = angular_acceleration[i];
    }
    for i in 0..4
    {
      derivative[QUATERNION_OFFSET + i] = 0.5 * quaternion_rate[i];
    }
    derivative
  }

  /// The right-hand side for a validated state (unit quaternion within
  /// [`Self::UNIT_QUATERNION_TOLERANCE`], renormalized first) and wrench.
  pub fn derivative(
    &self,
    state: &State,
    wrench: &Wrench) -> Result<StateDerivative, RigidBodyError>
  {
    let state = validated(state)?;
    wrench.validate()?;
    let derivative = self.derivative_array(&state.to_array(), wrench);
    if derivative.iter().any(|value| !value.is_finite())
    {
      return Err(RigidBodyError::NonFiniteResult);
    }
    let packed = State::from_array(&derivative);
    Ok(StateDerivative
    {
      position: packed.position,
      velocity: packed.velocity,
      quaternion: packed.quaternion,
      angular_velocity_body: packed.angular_velocity_body,
    })
  }

  /// `J ω`, body components.
  pub fn angular_momentum_body(
    &self,
    state: &State) -> Result<Vector3, RigidBodyError>
  {
    let state = validated(state)?;
    finite_or_error(matrix_vector_product(
      &self.inertia,
      &state.angular_velocity_body))
  }

  /// `R(q) J ω`, inertial components; constant without torque.
  pub fn angular_momentum_inertial(
    &self,
    state: &State) -> Result<Vector3, RigidBodyError>
  {
    let state = validated(state)?;
    let body = self.angular_momentum_body(&state)?;
    finite_or_error(matrix_vector_product(
      &rotation_matrix(&state.quaternion),
      &body))
  }

  /// `½ m v·v + ½ ω·J ω`; constant without forces and torque.
  pub fn kinetic_energy(&self, state: &State) -> Result<f64, RigidBodyError>
  {
    let state = validated(state)?;
    let energy = 0.5
      * (self.mass * dot(&state.velocity, &state.velocity)
        + dot(
          &state.angular_velocity_body,
          &self.angular_momentum_body(&state)?));
    if energy.is_finite() { Ok(energy) } else { Err(RigidBodyError::NonFiniteResult) }
  }

  /// `n` classical RK4 steps of size `time_step` under a fixed wrench, with
  /// the quaternion projected back to `|q| = 1` after every step.
  pub fn propagate_classical_4(
    &self,
    state: &State,
    wrench: &Wrench,
    time_step: f64,
    steps: usize) -> Result<State, RigidBodyError>
  {
    let state = validated(state)?;
    wrench.validate()?;
    if !time_step.is_finite() || time_step <= 0.0
    {
      return Err(RigidBodyError::NonPositiveTimeStep);
    }
    let right_hand_side =
      |_t: f64, y: &[f64; STATE_DIMENSION]| self.derivative_array(y, wrench);
    let mut y = state.to_array();
    for n in 0..steps
    {
      y = runge_kutta_4_step(&right_hand_side, n as f64 * time_step, &y, time_step);
      project_quaternion(&mut y)?;
      if y.iter().any(|value| !value.is_finite())
      {
        return Err(RigidBodyError::NonFiniteResult);
      }
    }
    Ok(State::from_array(&y))
  }
}

/// `|q|` of the packed state; 1 along the exact flow.
pub fn quaternion_norm(y: &[f64; STATE_DIMENSION]) -> f64
{
  algebra::quaternion_norm(&[y[6], y[7], y[8], y[9]])
}

/// Replace `q` by `q / |q|` in the packed state, keeping its sign (GNI §IV.4,
/// the standard projection onto the unit sphere). Fails, leaving `y`
/// unchanged, if `|q|` is zero or not finite.
pub fn project_quaternion(
  y: &mut [f64; STATE_DIMENSION]) -> Result<(), RigidBodyError>
{
  let projected = normalized_quaternion(&[y[6], y[7], y[8], y[9]])
    .ok_or(RigidBodyError::NonFiniteResult)?;
  y[QUATERNION_OFFSET..ANGULAR_VELOCITY_OFFSET].copy_from_slice(&projected);
  Ok(())
}

/// `I + (dt / 2) (hat(ω_mid) − hat(h_mid) J⁻¹)`, the Jacobian of the implicit
/// midpoint residual in the end-of-step angular momentum.
fn newton_jacobian(
  angular_velocity_midpoint: &Vector3,
  angular_momentum_midpoint: &Vector3,
  inverse_inertia: &Matrix3,
  half_step: f64) -> Matrix3
{
  let velocity_part = algebra::hat(angular_velocity_midpoint);
  let momentum_part =
    matrix_product(&algebra::hat(angular_momentum_midpoint), inverse_inertia);
  core::array::from_fn(|i| {
    core::array::from_fn(|j| {
      IDENTITY_MATRIX[i][j] + (velocity_part[i][j] - momentum_part[i][j]) * half_step
    })
  })
}
