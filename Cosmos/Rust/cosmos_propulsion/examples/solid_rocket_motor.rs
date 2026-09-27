//! Run a solid rocket motor / gas generator scenario and print its trajectory
//! as TSV (same columns as golden/solid_rocket_motor_<scenario>.tsv).
//!
//!   cargo run --release -p cosmos_propulsion --example solid_rocket_motor -- gas_generator
//!
//! Scenarios: huzel_cartridge, tubular_booster, gas_generator.

use cosmos_propulsion::solid_rocket_motor::scenarios::{
  cartridge_with_four_pintle_valves as four_valves, huzel_cartridge, tubular_booster,
};

fn print_rows<const N: usize>(header: &str, rows: impl Iterator<Item = [f64; N]>)
{
  println!("{header}");
  for row in rows
  {
    let fields: Vec<String> = row.iter().map(|x| format!("{x:e}")).collect();
    println!("{}", fields.join("\t"));
  }
}

const MOTOR_HEADER: &str = "time\tburned_web\tchamber_pressure\tburning_rate\tgeneration_rate\tnozzle_mass_flow\tthrust\texpelled_mass\ttotal_impulse\tburned_propellant_mass\tmass_residual";

fn main()
{
  let scenario = std::env::args().nth(1).unwrap_or_else(|| "gas_generator".to_string());
  match scenario.as_str()
  {
    "huzel_cartridge" =>
    {
      let samples = huzel_cartridge::motor().simulate(
        huzel_cartridge::INITIAL_PRESSURE,
        huzel_cartridge::STEP,
        huzel_cartridge::STEP_COUNT,
        huzel_cartridge::SAMPLE_INTERVAL,
      );
      print_rows(MOTOR_HEADER, samples.iter().map(|s| s.as_row()));
    }
    "tubular_booster" =>
    {
      let samples = tubular_booster::motor().simulate(
        tubular_booster::INITIAL_PRESSURE,
        tubular_booster::STEP,
        tubular_booster::STEP_COUNT,
        tubular_booster::SAMPLE_INTERVAL,
      );
      print_rows(MOTOR_HEADER, samples.iter().map(|s| s.as_row()));
    }
    "gas_generator" =>
    {
      let system = four_valves::system();
      let initial = system.initial_state(
        four_valves::INITIAL_CHAMBER_PRESSURE,
        four_valves::INITIAL_PLENUM_PRESSURE,
        four_valves::INITIAL_PLENUM_TEMPERATURE,
      );
      let samples =
        system.simulate(&initial, four_valves::STEP, four_valves::STEP_COUNT, four_valves::SAMPLE_INTERVAL);
      print_rows(
        "time\tburned_web\tchamber_pressure\tplenum_pressure\tplenum_temperature\tburning_rate\tgeneration_rate\torifice_mass_flow\tvalve_mass_flow\tthrust\texpelled_mass\ttotal_impulse\tburned_propellant_mass\tmass_residual",
        samples.iter().map(|s| s.as_row()),
      );
    }
    other =>
    {
      eprintln!("unknown scenario {other}; use huzel_cartridge, tubular_booster or gas_generator");
      std::process::exit(1);
    }
  }
}
