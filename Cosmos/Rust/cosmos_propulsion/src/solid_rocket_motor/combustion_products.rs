//! The combustion gas as a calorically perfect gas with fixed stagnation
//! temperature, molar mass and ratio of specific heats.
//!
//! R = R_u / M (Turns 3e Eq. 2.3, p. 13); Gamma (Hill & Peterson 2e Eq. 3.14,
//! p. 71); c* = sqrt(R T_0) / Gamma (Sutton 9e Eq. 3-32, p. 63). T_0 does not
//! depend on chamber pressure (Hill & Peterson p. 599).
//! C++ twin: `Propulsion/SolidRocketMotor/CombustionProducts.h`.

use cosmos_numerical::field::RealField;

use super::SolidRocketMotorError;

/// CODATA 2018 molar gas constant, J / (mol K).
pub const UNIVERSAL_GAS_CONSTANT_SI: f64 = 8.314462618;

/// Gamma(gamma) = sqrt(gamma) (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1))).
pub fn flow_function<T: RealField>(heat_capacity_ratio: T) -> T
{
  let gamma = heat_capacity_ratio;
  let one = T::one();
  let two = T::from_f64(2.0);
  gamma.square_root() * (two / (gamma + one)).power((gamma + one) / (two * (gamma - one)))
}

/// p_t / p_1 = (2 / (gamma + 1))^(gamma / (gamma - 1)), Sutton Eq. 3-20.
pub fn critical_pressure_ratio<T: RealField>(heat_capacity_ratio: T) -> T
{
  let gamma = heat_capacity_ratio;
  let one = T::one();
  let two = T::from_f64(2.0);
  (two / (gamma + one)).power(gamma / (gamma - one))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CombustionProducts<T: RealField>
{
  stagnation_temperature: T,
  molar_mass: T,
  heat_capacity_ratio: T,
  universal_gas_constant: T,
}

impl<T: RealField> CombustionProducts<T>
{
  pub fn new(
    stagnation_temperature: T,
    molar_mass: T,
    heat_capacity_ratio: T,
    universal_gas_constant: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(stagnation_temperature > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveTemperature);
    }
    if !(molar_mass > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveMolarMass);
    }
    if !(heat_capacity_ratio > T::one())
    {
      return Err(SolidRocketMotorError::HeatCapacityRatioNotAboveOne);
    }
    if !(universal_gas_constant > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveGasConstant);
    }
    Ok(Self { stagnation_temperature, molar_mass, heat_capacity_ratio, universal_gas_constant })
  }

  /// With the CODATA gas constant.
  pub fn with_si_gas_constant(
    stagnation_temperature: T,
    molar_mass: T,
    heat_capacity_ratio: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    Self::new(
      stagnation_temperature,
      molar_mass,
      heat_capacity_ratio,
      T::from_f64(UNIVERSAL_GAS_CONSTANT_SI),
    )
  }

  /// Invert c* = sqrt(R_u T_0 / M) / Gamma for M (Huzel & Huang p. 116 quote
  /// c* and T_0).
  pub fn from_characteristic_velocity(
    characteristic_velocity: T,
    stagnation_temperature: T,
    heat_capacity_ratio: T,
    universal_gas_constant: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(characteristic_velocity > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveCharacteristicVelocity);
    }
    if !(heat_capacity_ratio > T::one())
    {
      return Err(SolidRocketMotorError::HeatCapacityRatioNotAboveOne);
    }
    let gamma_c_star = flow_function(heat_capacity_ratio) * characteristic_velocity;
    Self::new(
      stagnation_temperature,
      universal_gas_constant * stagnation_temperature / (gamma_c_star * gamma_c_star),
      heat_capacity_ratio,
      universal_gas_constant,
    )
  }

  pub fn stagnation_temperature(&self) -> T { self.stagnation_temperature }
  pub fn molar_mass(&self) -> T { self.molar_mass }
  pub fn heat_capacity_ratio(&self) -> T { self.heat_capacity_ratio }

  /// R = R_u / M.
  pub fn specific_gas_constant(&self) -> T
  {
    self.universal_gas_constant / self.molar_mass
  }

  /// c_p = gamma R / (gamma - 1).
  pub fn specific_heat_at_constant_pressure(&self) -> T
  {
    self.heat_capacity_ratio * self.specific_gas_constant() / (self.heat_capacity_ratio - T::one())
  }

  /// c_v = R / (gamma - 1).
  pub fn specific_heat_at_constant_volume(&self) -> T
  {
    self.specific_gas_constant() / (self.heat_capacity_ratio - T::one())
  }

  pub fn flow_function_value(&self) -> T
  {
    flow_function(self.heat_capacity_ratio)
  }

  /// c* = sqrt(R T_0) / Gamma.
  pub fn characteristic_velocity(&self) -> T
  {
    (self.specific_gas_constant() * self.stagnation_temperature).square_root()
      / self.flow_function_value()
  }

  pub fn density_at(&self, pressure: T) -> T
  {
    pressure / (self.specific_gas_constant() * self.stagnation_temperature)
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn flow_function_and_critical_ratio_for_air()
  {
    assert!((flow_function(1.4_f64) - 0.684731).abs() < 1.0e-6);
    assert!((critical_pressure_ratio(1.4_f64) - 0.528282).abs() < 1.0e-6);
  }

  #[test]
  fn characteristic_velocity_round_trips()
  {
    let gas = CombustionProducts::from_characteristic_velocity(
      1298.448_f64,
      1672.039,
      1.25,
      UNIVERSAL_GAS_CONSTANT_SI,
    )
    .unwrap();
    assert!((gas.characteristic_velocity() - 1298.448).abs() < 1.0e-9);
  }

  #[test]
  fn rejects_nonphysical_gas()
  {
    assert_eq!(
      CombustionProducts::with_si_gas_constant(3000.0_f64, 0.029, 1.0),
      Err(SolidRocketMotorError::HeatCapacityRatioNotAboveOne)
    );
  }
}
