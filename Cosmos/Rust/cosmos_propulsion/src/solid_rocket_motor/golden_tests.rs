//! Cross-language golden vectors (story 16): the C++ reference emits
//! `golden/solid_rocket_motor_*.tsv` via
//! `tools/emit_solid_rocket_motor_golden.cpp`; these tests recompute every row
//! in Rust. Components agree to 1e-14 relative; trajectories, which integrate
//! tens of thousands of RK4 steps through pow/exp, to 1e-10 relative. Plus the
//! physics properties of the systems, stated independently of C++.

use super::burning_rate::{erosive_burning_rate, ErosiveBurning, TemperatureSensitivity};
use super::combustion_products::{critical_pressure_ratio, CombustionProducts, UNIVERSAL_GAS_CONSTANT_SI};
use super::compressible_flow::{ideal_thrust, restriction_mass_flow, supersonic_exit_mach};
use super::grain::{Grain, TubularGrain};
use super::pintle_valve::{OpeningSchedule, PintleValve};
use super::scenarios::{cartridge_with_four_pintle_valves as four_valves, huzel_cartridge, tubular_booster};

const COMPONENTS: &str = include_str!("../../../golden/solid_rocket_motor_components.tsv");
const HUZEL: &str = include_str!("../../../golden/solid_rocket_motor_huzel_cartridge.tsv");
const BOOSTER: &str = include_str!("../../../golden/solid_rocket_motor_tubular_booster.tsv");
const GAS_GENERATOR: &str = include_str!("../../../golden/solid_rocket_motor_gas_generator.tsv");

fn rows(table: &str) -> impl Iterator<Item = Vec<&str>>
{
  table.lines().filter(|l| !l.starts_with('#')).skip(1).map(|l| l.split('\t').collect())
}

fn numbers(fields: &[&str]) -> Vec<f64>
{
  fields.iter().map(|f| f.parse::<f64>().expect("golden number")).collect()
}

fn assert_close(actual: f64, expected: f64, relative: f64, absolute: f64, context: &str)
{
  let tolerance = absolute + relative * expected.abs();
  assert!(
    (actual - expected).abs() <= tolerance,
    "{context}: rust {actual:e} vs c++ {expected:e} (diff {:e})",
    (actual - expected).abs()
  );
}

#[test]
fn components_agree_with_the_cpp_twin()
{
  let mut checked = 0;
  for fields in rows(COMPONENTS)
  {
    let kind = fields[0];
    let x = numbers(&fields[1..9]);
    let expected = numbers(&fields[9..10])[0];
    let actual = match kind
    {
      "restriction" =>
      {
        let gas = CombustionProducts::with_si_gas_constant(2000.0, x[1], x[0]).unwrap();
        restriction_mass_flow(&gas, x[2], x[3], x[4], x[5], x[6])
      }
      "mach" => supersonic_exit_mach(x[1], x[0]),
      "thrust" =>
      {
        let gas = CombustionProducts::with_si_gas_constant(3000.0, x[1], x[0]).unwrap();
        ideal_thrust(&gas, x[3], x[4], x[5], x[6], x[2], x[7])
      }
      "erosive" =>
      {
        let erosive = ErosiveBurning::with_default_solver(x[1], x[2]).unwrap();
        erosive_burning_rate(x[0], &erosive, x[3], x[4], x[5], x[6])
      }
      "temperature" =>
      {
        let law = if x[0] < 0.5
        {
          TemperatureSensitivity::exponential(x[1], x[2]).unwrap()
        }
        else
        {
          TemperatureSensitivity::explosion_temperature(x[1], x[2]).unwrap()
        };
        law.coefficient_factor(x[3])
      }
      "tubular_area" | "tubular_volume" =>
      {
        let grain = TubularGrain::new(x[0], x[1], x[2], x[3] > 0.5, x[4]).unwrap();
        if kind == "tubular_area" { grain.burning_area(x[5]) } else { grain.gas_volume(x[5]) }
      }
      "pintle_area" =>
      {
        let valve = PintleValve::new(x[0], 4.0e-4, 0.95, OpeningSchedule::constant(1.0).unwrap()).unwrap();
        valve.flow_area(x[1])
      }
      other => panic!("unknown golden kind {other}"),
    };
    assert_close(actual, expected, 1.0e-14, 1.0e-300, kind);
    checked += 1;
  }
  assert!(checked >= 150, "components golden looks truncated: {checked} rows");
}

fn compare_trajectory(table: &str, computed: Vec<Vec<f64>>, name: &str)
{
  let expected: Vec<Vec<f64>> = rows(table).map(|f| numbers(&f)).collect();
  assert_eq!(computed.len(), expected.len(), "{name}: sample count");
  for (row, (a, e)) in computed.iter().zip(expected.iter()).enumerate()
  {
    assert_eq!(a.len(), e.len());
    for column in 0..a.len()
    {
      // Mass residuals are round-off themselves; compare them absolutely.
      let absolute = if column == a.len() - 1 { 1.0e-11 } else { 1.0e-9 };
      assert_close(a[column], e[column], 1.0e-10, absolute, &format!("{name} row {row} column {column}"));
    }
  }
}

#[test]
fn huzel_cartridge_trajectory_agrees_with_the_cpp_twin()
{
  let samples = huzel_cartridge::motor().simulate(
    huzel_cartridge::INITIAL_PRESSURE,
    huzel_cartridge::STEP,
    huzel_cartridge::STEP_COUNT,
    huzel_cartridge::SAMPLE_INTERVAL,
  );
  compare_trajectory(HUZEL, samples.iter().map(|s| s.as_row().to_vec()).collect(), "huzel");
}

#[test]
fn tubular_booster_trajectory_agrees_with_the_cpp_twin()
{
  let samples = tubular_booster::motor().simulate(
    tubular_booster::INITIAL_PRESSURE,
    tubular_booster::STEP,
    tubular_booster::STEP_COUNT,
    tubular_booster::SAMPLE_INTERVAL,
  );
  compare_trajectory(BOOSTER, samples.iter().map(|s| s.as_row().to_vec()).collect(), "booster");
}

#[test]
fn gas_generator_trajectory_agrees_with_the_cpp_twin()
{
  let system = four_valves::system();
  let initial = system.initial_state(
    four_valves::INITIAL_CHAMBER_PRESSURE,
    four_valves::INITIAL_PLENUM_PRESSURE,
    four_valves::INITIAL_PLENUM_TEMPERATURE,
  );
  let samples = system.simulate(&initial, four_valves::STEP, four_valves::STEP_COUNT, four_valves::SAMPLE_INTERVAL);
  compare_trajectory(GAS_GENERATOR, samples.iter().map(|s| s.as_row().to_vec()).collect(), "gas generator");
}

fn nearest<S: Copy, F: Fn(&S) -> f64>(samples: &[S], time: f64, time_of: F) -> S
{
  *samples
    .iter()
    .min_by(|a, b| (time_of(a) - time).abs().partial_cmp(&(time_of(b) - time).abs()).unwrap())
    .unwrap()
}

/// Huzel & Huang p. 116 within 1.5%; the gap is the (rho_b - rho) filling
/// term (Hill & Peterson Eq. 12.28), which the quoted sizing ignores.
#[test]
fn huzel_cartridge_reproduces_the_quoted_operating_point()
{
  let motor = huzel_cartridge::motor();
  let samples = motor.simulate(huzel_cartridge::INITIAL_PRESSURE, huzel_cartridge::STEP, 12000, 500);
  let steady = nearest(&samples, 0.5, |s| s.time);
  let quoted_pressure = huzel_cartridge::CHAMBER_PRESSURE;
  assert!((steady.chamber_pressure - quoted_pressure).abs() < 0.015 * quoted_pressure);
  assert!((steady.nozzle_mass_flow - huzel_cartridge::MASS_FLOW).abs() < 0.015 * huzel_cartridge::MASS_FLOW);
  for s in &samples
  {
    assert!(s.mass_residual.abs() < 1.0e-9 * (1.0 + s.burned_propellant_mass));
  }
}

/// The tubular booster burns exactly its loaded propellant (the RK4 step lands
/// on burnout) and its progressive grain gives a rising pressure trace.
#[test]
fn tubular_booster_burns_out_exactly()
{
  let motor = tubular_booster::motor();
  let samples = motor.simulate(
    tubular_booster::INITIAL_PRESSURE,
    tubular_booster::STEP,
    tubular_booster::STEP_COUNT,
    tubular_booster::SAMPLE_INTERVAL,
  );
  let last = samples.last().unwrap();
  let loaded = tubular_booster::PROPELLANT_DENSITY * motor.chamber().grain().propellant_volume(0.0);
  assert_eq!(last.burned_web, motor.chamber().grain().web());
  assert!((last.burned_propellant_mass - loaded).abs() < 1.0e-9 * loaded);
  assert!(nearest(&samples, 1.0, |s| s.time).chamber_pressure < nearest(&samples, 5.0, |s| s.time).chamber_pressure);
}

/// Sutton Fig. 12-27: closing valves raises the plenum pressure, unchokes
/// the orifice, and raises the grain-chamber pressure; T_plenum -> T_0.
#[test]
fn closing_pintle_valves_throttles_the_gas_generator()
{
  let system = four_valves::system();
  let initial = system.initial_state(
    four_valves::INITIAL_CHAMBER_PRESSURE,
    four_valves::INITIAL_PLENUM_PRESSURE,
    four_valves::INITIAL_PLENUM_TEMPERATURE,
  );
  let samples = system.simulate(&initial, four_valves::STEP, 19000, 500);
  let gamma = huzel_cartridge::HEAT_CAPACITY_RATIO;
  let open = nearest(&samples, 0.35, |s| s.time);
  let throttled = nearest(&samples, 0.90, |s| s.time);
  let t0 = huzel_cartridge::STAGNATION_TEMPERATURE;
  assert!((open.plenum_temperature - t0).abs() < 1.0e-6 * t0);
  assert!(open.plenum_pressure / open.chamber_pressure < critical_pressure_ratio(gamma));
  assert!(throttled.plenum_pressure > 1.8 * open.plenum_pressure);
  assert!(throttled.plenum_pressure / throttled.chamber_pressure > critical_pressure_ratio(gamma));
  assert!(throttled.chamber_pressure > 1.02 * open.chamber_pressure);
  for s in &samples
  {
    assert!(s.mass_residual.abs() < 1.0e-9 * (1.0 + s.burned_propellant_mass));
  }
  let _ = UNIVERSAL_GAS_CONSTANT_SI;
}
