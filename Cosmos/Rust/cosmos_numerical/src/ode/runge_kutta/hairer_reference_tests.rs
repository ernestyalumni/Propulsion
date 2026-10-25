//! Step-for-step comparison with Hairer's Fortran `DOPRI5` and `DOP853`.
//!
//! `tools/hairer_reference/emit_golden.sh` fetches the reference codes from
//! unige.ch (hash-checked, never vendored), runs them with dense output on
//! the Arenstorf orbit and the Prothero–Robinson problem, and records every
//! accepted step, dense output at θ = 1/4, 1/2, 3/4 of each step, and the
//! final counters. The Rust integrators must take the same steps: the same
//! number of attempts and acceptances, the same function-evaluation count,
//! the same outcome (completion, or the stiffness abort), and the same step
//! endpoints, states and dense values, bit for bit on x86-64 Linux (glibc
//! `pow`, `sqrt`, `sin`, `cos` on both sides).
//!
//! Bitwise agreement is the point: an embedded error estimate is a
//! difference of nearly equal stage sums, so one unit in the last place in
//! any stage changes `err` by about 10⁻⁹ relative and the next step size by
//! about 10⁻¹⁰; any deviation in operation order shows up within a few
//! steps. That is why `stage_input` forms single-term rows as `y + (h a) k`
//! and `PIStepSizeControl` divides `h` by the clamped reciprocal scale, as
//! the Fortran does.

use super::dormand_prince_5::DormandPrince5;
use super::dormand_prince_853::DormandPrince853;
use super::embedded::{
  EmbeddedPair, EmbeddedRungeKutta, IntegrationError, IntegratorOptions,
  StiffnessPolicy,
};
use super::test_problems::{arenstorf, prothero_robinson};
use super::tolerance::Tolerance;

/// The largest accepted `|a − b| / max(1, |b|)` over every compared number:
/// zero where both sides use glibc's libm, a looser bound elsewhere.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const AGREEMENT: f64 = 0.0;
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
const AGREEMENT: f64 = 1.0e-9;

struct StepRow
{
  number: usize,
  start: f64,
  end: f64,
  state: Vec<f64>,
}

struct DenseRow
{
  number: usize,
  time: f64,
  state: Vec<f64>,
}

struct Golden
{
  name: &'static str,
  start: f64,
  end: f64,
  initial_state: Vec<f64>,
  relative_tolerance: Vec<f64>,
  absolute_tolerance: Vec<f64>,
  maximum_step: f64,
  initial_step: f64,
  steps: Vec<StepRow>,
  dense: Vec<DenseRow>,
  outcome: i64,
  function_evaluations: usize,
  attempts: usize,
  accepted: usize,
  final_time: f64,
  final_step: f64,
}

fn numbers(fields: &[&str]) -> Vec<f64>
{
  fields.iter().map(|field| field.trim().parse::<f64>().unwrap()).collect()
}

fn parse(name: &'static str, text: &str) -> Golden
{
  let mut golden = Golden {
    name,
    start: 0.0,
    end: 0.0,
    initial_state: Vec::new(),
    relative_tolerance: Vec::new(),
    absolute_tolerance: Vec::new(),
    maximum_step: 0.0,
    initial_step: 0.0,
    steps: Vec::new(),
    dense: Vec::new(),
    outcome: 0,
    function_evaluations: 0,
    attempts: 0,
    accepted: 0,
    final_time: 0.0,
    final_step: 0.0,
  };
  for line in text.lines()
  {
    let fields: Vec<&str> = line.split('\t').collect();
    match fields[0]
    {
      "# x0" =>
      {
        golden.start = numbers(&fields[1..2])[0];
        golden.end = numbers(&fields[3..4])[0];
      }
      "# y0" => golden.initial_state = numbers(&fields[1..]),
      "# rtol" => golden.relative_tolerance = numbers(&fields[1..]),
      "# atol" => golden.absolute_tolerance = numbers(&fields[1..]),
      "# work(1)_uround" =>
      {
        golden.maximum_step = numbers(&fields[3..4])[0];
        golden.initial_step = numbers(&fields[5..6])[0];
      }
      "step" =>
      {
        let values = numbers(&fields[1..]);
        golden.steps.push(StepRow {
          number: values[0] as usize,
          start: values[1],
          end: values[2],
          state: values[3..].to_vec(),
        });
      }
      "dense" =>
      {
        let values = numbers(&fields[1..]);
        golden.dense.push(DenseRow {
          number: values[0] as usize,
          time: values[2],
          state: values[3..].to_vec(),
        });
      }
      "stats" =>
      {
        let values = numbers(&fields[1..]);
        golden.outcome = values[0] as i64;
        golden.function_evaluations = values[1] as usize;
        golden.attempts = values[2] as usize;
        golden.accepted = values[3] as usize;
        golden.final_time = values[5];
        golden.final_step = values[6];
      }
      _ => {}
    }
  }
  assert!(!golden.steps.is_empty(), "{name}: no steps parsed");
  golden
}

fn difference(actual: f64, expected: f64) -> f64
{
  (actual - expected).abs() / expected.abs().max(1.0)
}

fn to_array<const N: usize>(values: &[f64]) -> [f64; N]
{
  values.try_into().unwrap()
}

/// Runs one golden case and returns the largest difference seen.
fn reproduce<const N: usize, M>(
  golden: &Golden,
  rhs: fn(f64, &[f64; N]) -> [f64; N]) -> f64
where
  M: EmbeddedPair<f64, N> + Default,
{
  let name = golden.name;
  let tolerance = if golden.relative_tolerance.len() == 1
  {
    Tolerance::uniform(golden.absolute_tolerance[0], golden.relative_tolerance[0])
  }
  else
  {
    Tolerance::componentwise(
      to_array(&golden.absolute_tolerance), to_array(&golden.relative_tolerance))
  }
  .unwrap();
  let options = IntegratorOptions {
    initial_step: (golden.initial_step != 0.0)
      .then_some(golden.initial_step.abs()),
    maximum_step: (golden.maximum_step != 0.0).then_some(golden.maximum_step),
    stiffness_policy: StiffnessPolicy::Abort,
    unit_roundoff: f64::EPSILON,
    ..IntegratorOptions::default()
  };
  let mut integration = EmbeddedRungeKutta::new(
    M::default(),
    rhs,
    golden.start,
    to_array(&golden.initial_state),
    golden.end,
    tolerance,
    options)
  .unwrap();

  let mut largest = 0.0_f64;
  let mut dense_rows = golden.dense.iter().peekable();
  let mut outcome = 1;
  for row in golden.steps.iter().skip(1)
  {
    let step = match integration.step()
    {
      Ok(Some(step)) => step,
      Ok(None) => panic!("{name}: finished before step {}", row.number),
      Err(IntegrationError::Stiff { .. }) =>
      {
        outcome = -4;
        break;
      }
      Err(error) => panic!("{name}: {error} at step {}", row.number),
    };
    largest = largest
      .max(difference(step.start_time, row.start))
      .max(difference(step.end_time, row.end));
    for (actual, expected) in integration.state().iter().zip(&row.state)
    {
      largest = largest.max(difference(*actual, *expected));
    }
    while let Some(dense) = dense_rows.next_if(|dense| dense.number == row.number)
    {
      let value = integration.dense_output(dense.time).unwrap();
      for (actual, expected) in value.iter().zip(&dense.state)
      {
        largest = largest.max(difference(*actual, *expected));
      }
    }
  }
  if outcome == 1
  {
    match integration.step()
    {
      Ok(None) => {}
      Err(IntegrationError::Stiff { .. }) => outcome = -4,
      other => panic!("{name}: expected the end, got {other:?}"),
    }
  }

  let statistics = integration.statistics();
  let reference_evaluations = golden.function_evaluations
    - usize::from(golden.initial_step != 0.0);
  assert_eq!(outcome, golden.outcome, "{name}: outcome");
  assert_eq!(statistics.attempts, golden.attempts, "{name}: attempts");
  assert_eq!(statistics.accepted, golden.accepted, "{name}: accepted");
  assert_eq!(
    statistics.function_evaluations, reference_evaluations, "{name}: nfcn");
  largest = largest
    .max(difference(integration.time(), golden.final_time))
    .max(difference(integration.step_size(), golden.final_step));
  assert!(largest <= AGREEMENT, "{name}: largest difference {largest:e}");
  largest
}

macro_rules! golden
{
  ($file:literal) =>
  {
    parse(
      $file,
      include_str!(concat!("../../../../golden/hairer_reference/", $file)))
  };
}

#[test]
fn dopri5_takes_the_steps_of_the_reference_code()
{
  let cases = [
    golden!("dopri5_arenstorf_tol1e-4.tsv"),
    golden!("dopri5_arenstorf_tol1e-7.tsv"),
    golden!("dopri5_arenstorf_tol1e-10.tsv"),
    golden!("dopri5_arenstorf_backward_tol1e-7.tsv"),
    golden!("dopri5_arenstorf_componentwise.tsv"),
    golden!("dopri5_arenstorf_given_step.tsv"),
  ];
  for case in &cases
  {
    let largest = reproduce::<4, DormandPrince5<f64, 4>>(case, arenstorf);
    eprintln!("{}: largest difference {largest:e}", case.name);
  }
  let stiff = golden!("dopri5_prothero_robinson_stiff.tsv");
  let largest = reproduce::<1, DormandPrince5<f64, 1>>(&stiff, prothero_robinson);
  eprintln!("{}: largest difference {largest:e}", stiff.name);
}

#[test]
fn dop853_takes_the_steps_of_the_reference_code()
{
  let cases = [
    golden!("dop853_arenstorf_tol1e-4.tsv"),
    golden!("dop853_arenstorf_tol1e-7.tsv"),
    golden!("dop853_arenstorf_tol1e-10.tsv"),
    golden!("dop853_arenstorf_backward_tol1e-7.tsv"),
    golden!("dop853_arenstorf_componentwise.tsv"),
    golden!("dop853_arenstorf_given_step.tsv"),
  ];
  for case in &cases
  {
    let largest = reproduce::<4, DormandPrince853<f64, 4>>(case, arenstorf);
    eprintln!("{}: largest difference {largest:e}", case.name);
  }
  let stiff = golden!("dop853_prothero_robinson_stiff.tsv");
  let largest =
    reproduce::<1, DormandPrince853<f64, 1>>(&stiff, prothero_robinson);
  eprintln!("{}: largest difference {largest:e}", stiff.name);
}
