//! Solid rocket motor internal ballistics, and the same grain run as a gas
//! generator feeding a plenum throttled by N pintle valves.
//!
//! Derivation: `documents/derivations/SolidRocketMotorGasGenerator.md` and
//! `documents/notes/topics/solid-ballistics.tex`. C++ twin:
//! `Cosmos/Source/Propulsion/SolidRocketMotor/`. Every function here mirrors
//! its C++ twin operation for operation, so the golden vectors in
//! `golden/solid_rocket_motor_*.tsv` agree to round-off.

pub mod burning_rate;
pub mod combustion_products;
pub mod compressible_flow;
pub mod gas_generator;
pub mod grain;
pub mod grain_chamber;
pub mod motor;
pub mod pintle_valve;
pub mod runge_kutta_4;
pub mod scenarios;

#[cfg(test)]
mod golden_tests;

/// A parameter outside its physical or mathematical domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolidRocketMotorError
{
  NonPositiveTemperature,
  NonPositiveMolarMass,
  HeatCapacityRatioNotAboveOne,
  NonPositiveGasConstant,
  NonPositiveCharacteristicVelocity,
  ExplosionTemperatureNotAboveReference,
  NonPositiveBurningRateCoefficient,
  NegativePressureExponent,
  NegativeErosiveParameter,
  NonPositiveTolerance,
  InvalidGrainDimensions,
  NegativeFreeVolume,
  NonPositiveDensity,
  NonPositiveArea,
  ExitAreaBelowThroatArea,
  DischargeCoefficientOutOfRange,
  NegativeAmbientPressure,
  EmptySchedule,
  OpeningOutOfRange,
  KnotTimesNotIncreasing,
  NonPositiveVolume,
  NegativeHeatConductance,
}
