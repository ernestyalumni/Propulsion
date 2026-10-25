//! The implicit midpoint step against the fixtures emitted by its mathphysics
//! twin (`Rust/lie-groups`, example `rigid_body_fixtures`). Both sides
//! accumulate every sum in the same order, so agreement is expected to the
//! last bit; the assertion allows `GOLDEN_RELATIVE_TOLERANCE` for a platform
//! whose libm `hypot`/`sqrt` rounds differently.

use super::*;

const GOLDEN: &str =
  include_str!("../../../golden/rigid_body_mathphysics_fixtures.tsv");

const GOLDEN_RELATIVE_TOLERANCE: f64 = 1.0e-13;
const COLUMNS: usize = 52;

fn relative_difference(a: f64, b: f64) -> f64
{
  let scale = a.abs().max(b.abs());
  if scale == 0.0 { 0.0 } else { (a - b).abs() / scale.max(1.0e-300) }
}

#[test]
fn implicit_midpoint_reproduces_the_mathphysics_fixtures()
{
  let mut checked = 0usize;
  let mut largest_difference: f64 = 0.0;
  let mut bitwise_rows = 0usize;
  for line in GOLDEN.lines()
  {
    if line.starts_with('#') || line.starts_with("name")
    {
      continue;
    }
    let fields: Vec<&str> = line.split('\t').collect();
    assert_eq!(fields.len(), COLUMNS, "malformed golden line: {line}");
    let value = |i: usize| fields[i].parse::<f64>().unwrap();
    let vector = |start: usize| [value(start), value(start + 1), value(start + 2)];

    let name = fields[0];
    let mass = value(1);
    let inertia = [vector(2), vector(5), vector(8)];
    let time_step = value(11);
    let steps: usize = fields[12].parse().unwrap();
    let mut initial = [0.0; STATE_DIMENSION];
    for (i, slot) in initial.iter_mut().enumerate()
    {
      *slot = value(13 + i);
    }
    let wrench = Wrench
    {
      force_body: vector(26),
      torque_body: vector(29),
      force_inertial: vector(32),
    };
    let mut expected = [0.0; STATE_DIMENSION];
    for (i, slot) in expected.iter_mut().enumerate()
    {
      *slot = value(35 + i);
    }
    let expected_energy = value(48);
    let expected_momentum = vector(49);

    let model = Model::new(mass, inertia).unwrap();
    let midpoint = ImplicitMidpoint::default();
    let mut state = State::from_array(&initial);
    for _ in 0..steps
    {
      state = midpoint.step(&model, &state, time_step, &wrench).unwrap().state;
    }
    let actual = state.to_array();
    let energy = model.kinetic_energy(&state).unwrap();
    let momentum = model.angular_momentum_inertial(&state).unwrap();

    let mut row_bitwise = actual == expected && energy == expected_energy;
    for i in 0..STATE_DIMENSION
    {
      let difference = relative_difference(actual[i], expected[i]);
      largest_difference = largest_difference.max(difference);
      assert!(
        difference <= GOLDEN_RELATIVE_TOLERANCE,
        "{name}: component {i}: rust {:.17e} vs mathphysics {:.17e}",
        actual[i],
        expected[i]);
    }
    assert!(relative_difference(energy, expected_energy) <= GOLDEN_RELATIVE_TOLERANCE);
    for i in 0..3
    {
      row_bitwise &= momentum[i] == expected_momentum[i];
      assert!(
        relative_difference(momentum[i], expected_momentum[i])
          <= GOLDEN_RELATIVE_TOLERANCE,
        "{name}: H_I[{i}]");
    }
    bitwise_rows += usize::from(row_bitwise);
    checked += 1;
  }
  println!(
    "{checked} fixtures, {bitwise_rows} bitwise identical, largest relative \
     difference {largest_difference:.3e}");
  assert_eq!(checked, 8, "golden file looks truncated");
}
