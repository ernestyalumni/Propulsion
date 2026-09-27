//! Named scenarios, mirroring `Cosmos/Source/Propulsion/SolidRocketMotor/
//! Scenarios.h` with the same arithmetic; change both together. Numbers are
//! quoted from a book (with the page) or labeled assumed / illustrative.

use std::f64::consts::PI;

use super::burning_rate::{ErosiveBurning, SaintRobertBurningRate, TemperatureSensitivity};
use super::combustion_products::{CombustionProducts, UNIVERSAL_GAS_CONSTANT_SI};
use super::gas_generator::{GasGeneratorValveSystem, Plenum};
use super::grain::{EndBurningGrain, TubularGrain};
use super::grain_chamber::GrainChamber;
use super::motor::SolidRocketMotor;
use super::pintle_valve::{Knot, OpeningSchedule, PintleValve};

pub const STANDARD_ATMOSPHERE_PRESSURE: f64 = 101325.0;

/// Huzel & Huang (1992), p. 116: solid-propellant start cartridge. Quoted:
/// 1000 psia, 4.7 lb/s, 2550 F, c* = 4260 ft/s, ~1.0 s.
pub mod huzel_cartridge
{
  use super::*;

  pub const PASCALS_PER_PSI: f64 = 6894.757293168;
  pub const KILOGRAMS_PER_POUND: f64 = 0.45359237;
  pub const METERS_PER_FOOT: f64 = 0.3048;

  pub const CHAMBER_PRESSURE: f64 = 1000.0 * PASCALS_PER_PSI;
  pub const MASS_FLOW: f64 = 4.7 * KILOGRAMS_PER_POUND;
  pub const CHARACTERISTIC_VELOCITY: f64 = 4260.0 * METERS_PER_FOOT;
  pub const STAGNATION_TEMPERATURE: f64 = (2550.0 - 32.0) * 5.0 / 9.0 + 273.15;
  pub const BURN_TIME: f64 = 1.0;

  // Assumed (not given on p. 116):
  pub const HEAT_CAPACITY_RATIO: f64 = 1.25;
  pub const PROPELLANT_DENSITY: f64 = 1600.0;
  pub const BURNING_RATE_AT_DESIGN: f64 = 0.02;
  pub const PRESSURE_EXPONENT: f64 = 0.4;
  pub const SIGMA_P: f64 = 0.0018;
  pub const REFERENCE_TEMPERATURE: f64 = 294.15;
  pub const FREE_VOLUME: f64 = 0.0015;

  pub const INITIAL_PRESSURE: f64 = 1.0e6;
  pub const STEP: f64 = 5.0e-5;
  pub const STEP_COUNT: usize = 26000;
  pub const SAMPLE_INTERVAL: usize = 500;

  pub fn products() -> CombustionProducts<f64>
  {
    CombustionProducts::from_characteristic_velocity(
      CHARACTERISTIC_VELOCITY,
      STAGNATION_TEMPERATURE,
      HEAT_CAPACITY_RATIO,
      UNIVERSAL_GAS_CONSTANT_SI,
    )
    .unwrap()
  }

  /// A_t = m_dot c* / p_1.
  pub fn throat_area() -> f64
  {
    MASS_FLOW * CHARACTERISTIC_VELOCITY / CHAMBER_PRESSURE
  }

  /// A_b = m_dot / (rho_b r).
  pub fn grain_radius() -> f64
  {
    let burning_area = MASS_FLOW / (PROPELLANT_DENSITY * BURNING_RATE_AT_DESIGN);
    (burning_area / PI).sqrt()
  }

  pub fn reference_coefficient() -> f64
  {
    BURNING_RATE_AT_DESIGN / CHAMBER_PRESSURE.powf(PRESSURE_EXPONENT)
  }

  pub fn chamber() -> GrainChamber<f64, EndBurningGrain<f64>>
  {
    GrainChamber::new(
      EndBurningGrain::new(grain_radius(), BURNING_RATE_AT_DESIGN * BURN_TIME, FREE_VOLUME).unwrap(),
      PROPELLANT_DENSITY,
      SaintRobertBurningRate::new(
        reference_coefficient(),
        PRESSURE_EXPONENT,
        TemperatureSensitivity::exponential(SIGMA_P, REFERENCE_TEMPERATURE).unwrap(),
      )
      .unwrap(),
      None,
      products(),
      REFERENCE_TEMPERATURE,
    )
    .unwrap()
  }

  pub fn motor() -> SolidRocketMotor<f64, EndBurningGrain<f64>>
  {
    SolidRocketMotor::new(chamber(), throat_area(), throat_area(), 1.0, STANDARD_ATMOSPHERE_PRESSURE).unwrap()
  }
}

/// Illustrative tubular booster segment, erosive burning, Williams law.
pub mod tubular_booster
{
  use super::*;

  pub const INNER_RADIUS: f64 = 0.05;
  pub const OUTER_RADIUS: f64 = 0.10;
  pub const LENGTH: f64 = 1.0;
  pub const FREE_VOLUME: f64 = 0.003;
  pub const PROPELLANT_DENSITY: f64 = 1760.0;
  pub const REFERENCE_PRESSURE: f64 = 7.0e6;
  pub const BURNING_RATE_AT_REFERENCE: f64 = 0.007;
  pub const PRESSURE_EXPONENT: f64 = 0.35;
  pub const REFERENCE_TEMPERATURE: f64 = 294.15;
  pub const EXPLOSION_TEMPERATURE: f64 = 794.15;
  pub const GRAIN_TEMPERATURE: f64 = 283.15;
  pub const EROSIVE_ALPHA: f64 = 2.0e-5;
  pub const EROSIVE_BETA: f64 = 53.0;
  pub const STAGNATION_TEMPERATURE: f64 = 3300.0;
  pub const MOLAR_MASS: f64 = 0.029;
  pub const HEAT_CAPACITY_RATIO: f64 = 1.18;
  pub const THROAT_AREA: f64 = 1.05e-3;
  pub const EXIT_AREA: f64 = 8.4e-3;
  pub const NOZZLE_DISCHARGE_COEFFICIENT: f64 = 0.98;

  pub const INITIAL_PRESSURE: f64 = 1.0e6;
  pub const STEP: f64 = 2.0e-4;
  pub const STEP_COUNT: usize = 45000;
  pub const SAMPLE_INTERVAL: usize = 1000;

  pub fn reference_coefficient() -> f64
  {
    BURNING_RATE_AT_REFERENCE / REFERENCE_PRESSURE.powf(PRESSURE_EXPONENT)
  }

  pub fn chamber() -> GrainChamber<f64, TubularGrain<f64>>
  {
    GrainChamber::new(
      TubularGrain::new(INNER_RADIUS, OUTER_RADIUS, LENGTH, true, FREE_VOLUME).unwrap(),
      PROPELLANT_DENSITY,
      SaintRobertBurningRate::new(
        reference_coefficient(),
        PRESSURE_EXPONENT,
        TemperatureSensitivity::explosion_temperature(EXPLOSION_TEMPERATURE, REFERENCE_TEMPERATURE).unwrap(),
      )
      .unwrap(),
      Some(ErosiveBurning::with_default_solver(EROSIVE_ALPHA, EROSIVE_BETA).unwrap()),
      CombustionProducts::with_si_gas_constant(STAGNATION_TEMPERATURE, MOLAR_MASS, HEAT_CAPACITY_RATIO).unwrap(),
      GRAIN_TEMPERATURE,
    )
    .unwrap()
  }

  pub fn motor() -> SolidRocketMotor<f64, TubularGrain<f64>>
  {
    SolidRocketMotor::new(
      chamber(),
      THROAT_AREA,
      EXIT_AREA,
      NOZZLE_DISCHARGE_COEFFICIENT,
      STANDARD_ATMOSPHERE_PRESSURE,
    )
    .unwrap()
  }
}

/// The Huzel cartridge as a gas generator into a plenum with four pintle
/// valves; valves 2 and 3 close between 0.40 s and 0.45 s.
pub mod cartridge_with_four_pintle_valves
{
  use super::*;

  pub const PLENUM_VOLUME: f64 = 0.004;
  pub const WALL_HEAT_CONDUCTANCE: f64 = 0.0;
  pub const WALL_TEMPERATURE: f64 = 300.0;
  pub const VALVE_THROAT_RADIUS: f64 = 0.010;
  pub const VALVE_EXPANSION_RATIO_FULL_OPEN: f64 = 4.0;
  pub const VALVE_DISCHARGE_COEFFICIENT: f64 = 0.95;
  pub const CLOSE_START: f64 = 0.40;
  pub const CLOSE_END: f64 = 0.45;
  pub const VALVE_COUNT: usize = 4;

  pub const INITIAL_CHAMBER_PRESSURE: f64 = 1.0e6;
  pub const INITIAL_PLENUM_PRESSURE: f64 = STANDARD_ATMOSPHERE_PRESSURE;
  pub const INITIAL_PLENUM_TEMPERATURE: f64 = WALL_TEMPERATURE;
  pub const STEP: f64 = 5.0e-5;
  pub const STEP_COUNT: usize = 26000;
  pub const SAMPLE_INTERVAL: usize = 500;

  pub fn valves() -> Vec<PintleValve<f64>>
  {
    let full_open_area = PI * VALVE_THROAT_RADIUS * VALVE_THROAT_RADIUS;
    (0..VALVE_COUNT)
      .map(|i| {
        let schedule = if i < 2
        {
          OpeningSchedule::constant(1.0).unwrap()
        }
        else
        {
          OpeningSchedule::new(vec![
            Knot { time: CLOSE_START, opening: 1.0 },
            Knot { time: CLOSE_END, opening: 0.0 },
          ])
          .unwrap()
        };
        PintleValve::new(
          VALVE_THROAT_RADIUS,
          VALVE_EXPANSION_RATIO_FULL_OPEN * full_open_area,
          VALVE_DISCHARGE_COEFFICIENT,
          schedule,
        )
        .unwrap()
      })
      .collect()
  }

  pub fn system() -> GasGeneratorValveSystem<f64, EndBurningGrain<f64>>
  {
    GasGeneratorValveSystem::new(
      huzel_cartridge::chamber(),
      huzel_cartridge::throat_area(),
      1.0,
      Plenum::new(PLENUM_VOLUME, WALL_HEAT_CONDUCTANCE, WALL_TEMPERATURE).unwrap(),
      valves(),
      STANDARD_ATMOSPHERE_PRESSURE,
    )
    .unwrap()
  }
}
