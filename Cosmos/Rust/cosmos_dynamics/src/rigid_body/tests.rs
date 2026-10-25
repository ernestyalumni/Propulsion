use super::algebra::{dot, Matrix3, Vector3};
use super::*;

fn diagonal(moments: Vector3) -> Matrix3
{
  [[moments[0], 0.0, 0.0], [0.0, moments[1], 0.0], [0.0, 0.0, moments[2]]]
}

fn norm(v: &Vector3) -> f64
{
  dot(v, v).sqrt()
}

fn spinning(angular_velocity_body: Vector3) -> State
{
  State { angular_velocity_body, ..State::default() }
}

/// Axisymmetric, torque free, J = diag(I1, I1, I3): ω3 is constant and
/// (ω1, ω2) turns at Ω = (I3 − I1) ω3 / I1, so
/// ω1 = a cos Ωt − b sin Ωt, ω2 = a sin Ωt + b cos Ωt.
fn axisymmetric_closed_form(
  transverse_moment: f64,
  axial_moment: f64,
  initial: Vector3,
  time: f64) -> Vector3
{
  let rate = (axial_moment - transverse_moment) * initial[2] / transverse_moment;
  let (sine, cosine) = (rate * time).sin_cos();
  [
    initial[0] * cosine - initial[1] * sine,
    initial[0] * sine + initial[1] * cosine,
    initial[2],
  ]
}

fn observed_order(coarse_error: f64, fine_error: f64) -> f64
{
  (coarse_error / fine_error).log2()
}

#[test]
fn axisymmetric_closed_form_is_matched_at_second_and_fourth_order()
{
  let (transverse, axial) = (2.0, 3.0);
  let model = Model::new(3.0, diagonal([transverse, transverse, axial])).unwrap();
  let initial = [0.4, -0.3, 1.2];
  let duration = 4.0;
  let exact = axisymmetric_closed_form(transverse, axial, initial, duration);
  let wrench = Wrench::default();
  let midpoint = ImplicitMidpoint::default();

  let midpoint_error = |steps: usize| {
    let time_step = duration / steps as f64;
    let mut state = spinning(initial);
    for _ in 0..steps
    {
      state = midpoint.step(&model, &state, time_step, &wrench).unwrap().state;
    }
    norm(&subtract(&state.angular_velocity_body, &exact))
  };
  let classical_error = |steps: usize| {
    let state = model
      .propagate_classical_4(
        &spinning(initial),
        &wrench,
        duration / steps as f64,
        steps)
      .unwrap();
    norm(&subtract(&state.angular_velocity_body, &exact))
  };

  let midpoint_order = observed_order(midpoint_error(200), midpoint_error(400));
  let classical_order = observed_order(classical_error(50), classical_error(100));
  println!("implicit midpoint order {midpoint_order:.4}, RK4 order {classical_order:.4}");
  assert!((midpoint_order - 2.0).abs() < 0.05, "{midpoint_order}");
  assert!((classical_order - 4.0).abs() < 0.1, "{classical_order}");
}

#[test]
fn implicit_midpoint_conserves_the_quadratic_invariants_and_rk4_drifts()
{
  let model = Model::new(1.0, diagonal([1.0, 2.0, 3.0])).unwrap();
  let initial = spinning([0.5, 0.6, 0.7]);
  let wrench = Wrench::default();
  let (time_step, steps) = (0.1, 10_000);
  let energy = model.kinetic_energy(&initial).unwrap();
  let momentum = norm(&model.angular_momentum_inertial(&initial).unwrap());

  let midpoint = ImplicitMidpoint::default();
  let mut state = initial;
  let mut midpoint_energy_drift: f64 = 0.0;
  let mut midpoint_momentum_drift: f64 = 0.0;
  for _ in 0..steps
  {
    state = midpoint.step(&model, &state, time_step, &wrench).unwrap().state;
    let e = model.kinetic_energy(&state).unwrap();
    let l = norm(&model.angular_momentum_inertial(&state).unwrap());
    midpoint_energy_drift = midpoint_energy_drift.max(((e - energy) / energy).abs());
    midpoint_momentum_drift =
      midpoint_momentum_drift.max(((l - momentum) / momentum).abs());
  }

  let classical = model
    .propagate_classical_4(&initial, &wrench, time_step, steps)
    .unwrap();
  let classical_energy_drift =
    ((model.kinetic_energy(&classical).unwrap() - energy) / energy).abs();
  let classical_momentum_drift = ((norm(
    &model.angular_momentum_inertial(&classical).unwrap())
    - momentum)
    / momentum)
    .abs();

  println!(
    "t = {}: implicit midpoint max |ΔE/E| = {midpoint_energy_drift:.3e}, \
     max |Δ|L|/|L|| = {midpoint_momentum_drift:.3e}; RK4 |ΔE/E| = \
     {classical_energy_drift:.3e}, |Δ|L|/|L|| = {classical_momentum_drift:.3e}",
    time_step * steps as f64);
  assert!(midpoint_energy_drift < 1.0e-12, "{midpoint_energy_drift}");
  assert!(midpoint_momentum_drift < 1.0e-12, "{midpoint_momentum_drift}");
  assert!(classical_energy_drift > 1.0e3 * midpoint_energy_drift.max(1.0e-16));
  assert!(classical_momentum_drift > 1.0e3 * midpoint_momentum_drift.max(1.0e-16));
}

/// Rotation about the intermediate principal axis is unstable: a spin
/// ω ≈ (ε, 1, ε) about axis 2 of J = diag(1, 2, 3) flips, ω2 → −ω2, after a
/// time of order ln(1/ε) / σ with σ = ω2 √((J2 − J1)(J3 − J2)/(J1 J3)).
#[test]
fn intermediate_axis_spin_flips()
{
  let model = Model::new(1.0, diagonal([1.0, 2.0, 3.0])).unwrap();
  let perturbation = 1.0e-3;
  let initial = spinning([perturbation, 1.0, perturbation]);
  let wrench = Wrench::default();
  let midpoint = ImplicitMidpoint::default();
  let (time_step, steps) = (0.01, 4_000);

  let mut state = initial;
  let mut first_flip_time = None;
  let mut smallest_axial_rate = f64::INFINITY;
  for n in 0..steps
  {
    state = midpoint.step(&model, &state, time_step, &wrench).unwrap().state;
    let axial_rate = state.angular_velocity_body[1];
    smallest_axial_rate = smallest_axial_rate.min(axial_rate);
    if first_flip_time.is_none() && axial_rate < 0.0
    {
      first_flip_time = Some((n + 1) as f64 * time_step);
    }
  }
  let growth_rate = (1.0_f64 * 1.0 / (1.0 * 3.0)).sqrt();
  let flip_time = first_flip_time.expect("ω2 never changed sign");
  println!(
    "first sign change of ω2 at t = {flip_time:.2} (ln(1/ε)/σ = {:.2}), \
     min ω2 = {smallest_axial_rate:.6}",
    (1.0 / perturbation).ln() / growth_rate);
  assert!(smallest_axial_rate < -0.99, "{smallest_axial_rate}");
  assert!(flip_time > 5.0 && flip_time < 30.0, "{flip_time}");
}

#[test]
fn derivative_and_derivative_array_agree_and_obey_the_equations()
{
  let inertia = [[2.0, 0.2, 0.1], [0.2, 3.0, 0.3], [0.1, 0.3, 4.0]];
  let model = Model::new(2.0, inertia).unwrap();
  let half_angle: f64 = 0.4;
  let state = State
  {
    position: [1.0, 2.0, 3.0],
    velocity: [0.3, -0.1, 0.2],
    quaternion: [half_angle.cos(), 0.0, half_angle.sin(), 0.0],
    angular_velocity_body: [0.3, -0.4, 0.8],
  };
  let wrench = Wrench
  {
    force_body: [1.0, 0.0, 0.0],
    torque_body: [0.1, -0.2, 0.05],
    force_inertial: [0.0, 0.0, -9.81],
  };
  let structured = model.derivative(&state, &wrench).unwrap();
  let packed = model.derivative_array(&state.to_array(), &wrench);
  assert_eq!(structured.to_array(), packed);
  assert_eq!(structured.position, state.velocity);

  // J ω' = τ − ω × J ω.
  let angular_momentum = matrix_vector_product(&inertia, &state.angular_velocity_body);
  let balance = subtract(
    &matrix_vector_product(&inertia, &structured.angular_velocity_body),
    &subtract(
      &wrench.torque_body,
      &cross(&state.angular_velocity_body, &angular_momentum)));
  assert!(norm(&balance) < 1.0e-14, "{balance:?}");

  // A rotation about y by 0.8 rad turns body x into (cos 0.8, 0, −sin 0.8).
  let angle = 2.0 * half_angle;
  let expected = [angle.cos() / 2.0, 0.0, -angle.sin() / 2.0 - 9.81 / 2.0];
  for i in 0..3
  {
    assert!((structured.velocity[i] - expected[i]).abs() < 1.0e-15);
  }
  // q' is tangent to the unit sphere: q · q' = 0.
  let tangency = (0..4)
    .map(|i| state.quaternion[i] * structured.quaternion[i])
    .sum::<f64>();
  assert!(tangency.abs() < 1.0e-16);
}

#[test]
fn packing_round_trips_and_projection_restores_the_unit_quaternion()
{
  let state = State
  {
    position: [1.0, 2.0, 3.0],
    velocity: [4.0, 5.0, 6.0],
    quaternion: [0.5, 0.5, 0.5, 0.5],
    angular_velocity_body: [7.0, 8.0, 9.0],
  };
  let mut y = state.to_array();
  assert_eq!(y, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 0.5, 0.5, 0.5, 7.0, 8.0, 9.0]);
  assert_eq!(State::from_array(&y), state);
  for i in 6..10
  {
    y[i] *= -3.0;
  }
  project_quaternion(&mut y).unwrap();
  assert!((quaternion_norm(&y) - 1.0).abs() < 1.0e-15);
  assert!((y[6] + 0.5).abs() < 1.0e-15, "the sign of the lift is kept");
  let mut zero = y;
  zero[6..10].copy_from_slice(&[0.0; 4]);
  assert_eq!(project_quaternion(&mut zero), Err(RigidBodyError::NonFiniteResult));
}

#[test]
fn rejects_unphysical_models_states_and_options()
{
  use RigidBodyError::*;
  let unit = diagonal([1.0, 1.0, 1.0]);
  assert_eq!(Model::new(0.0, unit).unwrap_err(), NonPositiveMass);
  assert_eq!(Model::new(f64::NAN, unit).unwrap_err(), NonFiniteInput);
  assert_eq!(Model::new(1.0, [[0.0; 3]; 3]).unwrap_err(), ZeroInertia);
  assert_eq!(
    Model::new(1.0, [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap_err(),
    InertiaNotSymmetric);
  assert_eq!(
    Model::new(1.0, diagonal([1.0, -1.0, 1.0])).unwrap_err(),
    InertiaNotPositiveDefinite);
  assert_eq!(
    Model::new(1.0, diagonal([1.0, 1.0, 3.0])).unwrap_err(),
    InertiaViolatesTriangleInequality);

  let model = Model::new(1.0, unit).unwrap();
  let not_unit = State { quaternion: [1.1, 0.0, 0.0, 0.0], ..State::default() };
  assert_eq!(model.kinetic_energy(&not_unit).unwrap_err(), QuaternionNotUnit);
  let midpoint = ImplicitMidpoint::default();
  assert_eq!(
    midpoint.step(&model, &State::default(), 0.0, &Wrench::default()).unwrap_err(),
    NonPositiveTimeStep);
  assert_eq!(ImplicitMidpoint::new(0.0, 10).unwrap_err(), InvalidNewtonTolerance);
  assert_eq!(ImplicitMidpoint::new(1.0e-3, 10).unwrap_err(), InvalidNewtonTolerance);
  assert_eq!(ImplicitMidpoint::new(1.0e-12, 0).unwrap_err(), InvalidNewtonIterationLimit);
  let bad_wrench = Wrench { torque_body: [f64::INFINITY, 0.0, 0.0], ..Wrench::default() };
  assert_eq!(model.derivative(&State::default(), &bad_wrench).unwrap_err(), NonFiniteInput);
}

#[test]
fn inverse_inertia_inverts_the_inertia()
{
  let inertia = [[2.0, 0.2, 0.1], [0.2, 3.0, 0.3], [0.1, 0.3, 4.0]];
  let model = Model::new(1.0, inertia).unwrap();
  let product = algebra::matrix_product(&inertia, model.inverse_inertia());
  for i in 0..3
  {
    for j in 0..3
    {
      assert!((product[i][j] - IDENTITY_MATRIX[i][j]).abs() < 1.0e-15);
    }
  }
}
