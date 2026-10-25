//! The implicit midpoint rule on the body angular momentum, a Cayley attitude
//! increment, and midpoint translation.
//!
//! # The step
//!
//! With `h = J ω` and the wrench held fixed over `[t, t + dt]`:
//!
//! ```text
//!   h₁ − h₀ = dt (τ − ω_m × h_m),   h_m = (h₀ + h₁)/2,   ω_m = J⁻¹ h_m
//! ```
//!
//! is the implicit midpoint rule (the one-stage Gauss method, HNW I §II.7;
//! GNI §VI.3) for Euler's equations, solved by Newton's method in `h₁`. It
//! conserves every quadratic invariant of the torque-free flow exactly
//! (GNI Theorem IV.2.1): `|h|²` and `½ h·J⁻¹h`. The attitude is advanced by
//! the Cayley increment
//!
//! ```text
//!   q₁ = q₀ ⊗ normalize(1, dt ω_m / 2),
//! ```
//!
//! a unit quaternion by construction, and the translation by the midpoint
//! rule with the body force rotated by the half-increment attitude `q_m`.
//! The method is second order and symmetric.

use super::algebra::{
  add, cross, hamilton_product, maximum_norm, matrix_vector_product,
  normalized_quaternion, rotation_matrix, scale, subtract, Matrix3, Vector3,
};
use super::{
  finite_or_error, newton_jacobian, validated, Model, RigidBodyError, State,
  Wrench,
};

/// Newton options for the implicit equation in the end-of-step momentum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImplicitMidpoint
{
  newton_tolerance: f64,
  maximum_newton_iterations: usize,
}

/// The state after one step, with the Newton diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImplicitMidpointStep
{
  pub state: State,
  pub newton_iterations: usize,
  /// Maximum-norm residual of the implicit equation at the accepted iterate.
  pub residual: f64,
}

impl Default for ImplicitMidpoint
{
  fn default() -> Self
  {
    Self
    {
      newton_tolerance: Self::DEFAULT_NEWTON_TOLERANCE,
      maximum_newton_iterations: Self::DEFAULT_MAXIMUM_NEWTON_ITERATIONS,
    }
  }
}

impl ImplicitMidpoint
{
  /// Relative to `1 + |h₀|_∞ + dt |τ|_∞`.
  pub const DEFAULT_NEWTON_TOLERANCE: f64 = 1.0e-12;
  pub const DEFAULT_MAXIMUM_NEWTON_ITERATIONS: usize = 20;
  /// Looser than this and the step is no longer the implicit midpoint rule.
  pub const LARGEST_NEWTON_TOLERANCE: f64 = 1.0e-4;
  pub const LARGEST_NEWTON_ITERATION_LIMIT: usize = 100;

  pub fn new(
    newton_tolerance: f64,
    maximum_newton_iterations: usize) -> Result<Self, RigidBodyError>
  {
    if !newton_tolerance.is_finite()
      || newton_tolerance <= 0.0
      || newton_tolerance > Self::LARGEST_NEWTON_TOLERANCE
    {
      return Err(RigidBodyError::InvalidNewtonTolerance);
    }
    if maximum_newton_iterations == 0
      || maximum_newton_iterations > Self::LARGEST_NEWTON_ITERATION_LIMIT
    {
      return Err(RigidBodyError::InvalidNewtonIterationLimit);
    }
    Ok(Self { newton_tolerance, maximum_newton_iterations })
  }

  pub fn newton_tolerance(&self) -> f64 { self.newton_tolerance }
  pub fn maximum_newton_iterations(&self) -> usize
  {
    self.maximum_newton_iterations
  }

  pub fn step(
    &self,
    model: &Model,
    state: &State,
    time_step: f64,
    wrench: &Wrench) -> Result<ImplicitMidpointStep, RigidBodyError>
  {
    let start = validated(state)?;
    wrench.validate()?;
    if !time_step.is_finite() || time_step <= 0.0
    {
      return Err(RigidBodyError::NonPositiveTimeStep);
    }

    let torque = wrench.torque_body;
    let initial_momentum = model.angular_momentum_body(&start)?;
    let mut final_momentum = add(
      &initial_momentum,
      &scale(
        &subtract(
          &torque,
          &cross(&start.angular_velocity_body, &initial_momentum)),
        time_step));
    let threshold = self.newton_tolerance
      * (1.0 + maximum_norm(&initial_momentum) + time_step * maximum_norm(&torque));
    if !threshold.is_finite()
    {
      return Err(RigidBodyError::NonFiniteResult);
    }

    let mut iteration = 0;
    let (midpoint_angular_velocity, residual) = loop
    {
      finite_or_error(final_momentum)?;
      let midpoint_momentum = scale(&add(&initial_momentum, &final_momentum), 0.5);
      let midpoint_angular_velocity = model.solve_inertia(&midpoint_momentum)?;
      let implicit_residual = subtract(
        &subtract(&final_momentum, &initial_momentum),
        &scale(
          &subtract(
            &torque,
            &cross(&midpoint_angular_velocity, &midpoint_momentum)),
          time_step));
      finite_or_error(implicit_residual)?;
      let residual = maximum_norm(&implicit_residual);
      if residual <= threshold
      {
        break (midpoint_angular_velocity, residual);
      }
      if iteration >= self.maximum_newton_iterations
      {
        return Err(RigidBodyError::NewtonDidNotConverge);
      }
      let jacobian = newton_jacobian(
        &midpoint_angular_velocity,
        &midpoint_momentum,
        model.inverse_inertia(),
        time_step / 2.0);
      let correction = solve_pivoted(jacobian, implicit_residual)?;
      final_momentum = subtract(&final_momentum, &correction);
      iteration += 1;
    };

    let increment = normalized_quaternion(&[
      1.0,
      time_step * midpoint_angular_velocity[0] / 2.0,
      time_step * midpoint_angular_velocity[1] / 2.0,
      time_step * midpoint_angular_velocity[2] / 2.0,
    ])
    .ok_or(RigidBodyError::NonFiniteResult)?;
    let half_scalar = ((1.0 + increment[0]) / 2.0).sqrt();
    let half_increment = [
      half_scalar,
      increment[1] / (2.0 * half_scalar),
      increment[2] / (2.0 * half_scalar),
      increment[3] / (2.0 * half_scalar),
    ];
    let midpoint_quaternion =
      normalized_quaternion(&hamilton_product(&start.quaternion, &half_increment))
        .ok_or(RigidBodyError::NonFiniteResult)?;

    let mut end = start;
    end.quaternion =
      normalized_quaternion(&hamilton_product(&start.quaternion, &increment))
        .ok_or(RigidBodyError::NonFiniteResult)?;
    end.angular_velocity_body = model.solve_inertia(&final_momentum)?;
    let force = add(
      &finite_or_error(matrix_vector_product(
        &rotation_matrix(&midpoint_quaternion),
        &wrench.force_body))?,
      &wrench.force_inertial);
    end.velocity = add(
      &start.velocity,
      &force.map(|component| (component / model.mass()) * time_step));
    end.position = add(
      &start.position,
      &scale(&add(&start.velocity, &end.velocity), time_step / 2.0));

    Ok(ImplicitMidpointStep
    {
      state: validated(&end)?,
      newton_iterations: iteration,
      residual,
    })
  }
}

/// `A x = b` by Gaussian elimination with partial pivoting; the Newton
/// Jacobian is not symmetric.
fn solve_pivoted(
  mut matrix: Matrix3,
  mut right_side: Vector3) -> Result<Vector3, RigidBodyError>
{
  for k in 0..3
  {
    let mut pivot = k;
    for i in k + 1..3
    {
      if matrix[i][k].abs() > matrix[pivot][k].abs()
      {
        pivot = i;
      }
    }
    if !matrix[pivot][k].is_finite() || matrix[pivot][k] == 0.0
    {
      return Err(RigidBodyError::SingularNewtonJacobian);
    }
    if pivot != k
    {
      matrix.swap(pivot, k);
      right_side.swap(pivot, k);
    }
    for i in k + 1..3
    {
      let multiplier = matrix[i][k] / matrix[k][k];
      for j in k + 1..3
      {
        matrix[i][j] -= multiplier * matrix[k][j];
      }
      right_side[i] -= multiplier * right_side[k];
    }
  }
  let mut solution = [0.0; 3];
  for i in (0..3).rev()
  {
    let mut value = right_side[i];
    for j in i + 1..3
    {
      value -= matrix[i][j] * solution[j];
    }
    solution[i] = value / matrix[i][i];
  }
  finite_or_error(solution)
}
