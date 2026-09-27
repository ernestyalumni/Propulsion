//! The lumped grain chamber. It holds gas mass m_c and burned web y; the
//! pressure is p = m_c R T_0 / V(y).
//!
//! Generation rho_b A_b r: Sutton 9e Eq. 12-1 (p. 444). Mass balance: Sutton
//! Eq. 12-3, Hill & Peterson Eq. 12.28 (p. 599), Humble Eq. 6.36 (p. 337).
//! C++ twin: `Propulsion/SolidRocketMotor/GrainChamber.h`.

use cosmos_numerical::field::RealField;

use super::burning_rate::{erosive_burning_rate, ErosiveBurning, SaintRobertBurningRate};
use super::combustion_products::CombustionProducts;
use super::grain::Grain;
use super::SolidRocketMotorError;

#[derive(Clone, Debug, PartialEq)]
pub struct GrainChamber<T: RealField, G: Grain<T>>
{
  grain: G,
  propellant_density: T,
  burning_rate_law: SaintRobertBurningRate<T>,
  erosive_burning: Option<ErosiveBurning<T>>,
  products: CombustionProducts<T>,
  grain_temperature: T,
}

impl<T: RealField, G: Grain<T>> GrainChamber<T, G>
{
  pub fn new(
    grain: G,
    propellant_density: T,
    burning_rate_law: SaintRobertBurningRate<T>,
    erosive_burning: Option<ErosiveBurning<T>>,
    products: CombustionProducts<T>,
    grain_temperature: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(propellant_density > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveDensity);
    }
    if !(grain_temperature > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveTemperature);
    }
    Ok(Self { grain, propellant_density, burning_rate_law, erosive_burning, products, grain_temperature })
  }

  pub fn grain(&self) -> &G { &self.grain }
  pub fn products(&self) -> &CombustionProducts<T> { &self.products }
  pub fn propellant_density(&self) -> T { self.propellant_density }
  pub fn burning_rate_law(&self) -> &SaintRobertBurningRate<T> { &self.burning_rate_law }
  pub fn grain_temperature(&self) -> T { self.grain_temperature }

  /// p = m_c R T_0 / V(y).
  pub fn pressure(&self, gas_mass: T, burned_web: T) -> T
  {
    gas_mass * self.products.specific_gas_constant() * self.products.stagnation_temperature()
      / self.grain.gas_volume(burned_web)
  }

  /// m_c = p V(y) / (R T_0).
  pub fn gas_mass_at(&self, pressure: T, burned_web: T) -> T
  {
    pressure * self.grain.gas_volume(burned_web)
      / (self.products.specific_gas_constant() * self.products.stagnation_temperature())
  }

  /// r(p, y), zero after burnout.
  pub fn burning_rate(&self, pressure: T, burned_web: T) -> T
  {
    let burning_area = self.grain.burning_area(burned_web);
    if burning_area <= T::zero()
    {
      return T::zero();
    }
    let base_rate = self.burning_rate_law.rate(pressure, self.grain_temperature);
    match &self.erosive_burning
    {
      None => base_rate,
      Some(erosive) => erosive_burning_rate(
        base_rate,
        erosive,
        self.propellant_density,
        burning_area,
        self.grain.port_area(burned_web),
        self.grain.port_hydraulic_diameter(burned_web),
      ),
    }
  }

  /// rho_b A_b r.
  pub fn generation_rate(&self, burning_rate: T, burned_web: T) -> T
  {
    self.propellant_density * self.grain.burning_area(burned_web) * burning_rate
  }
}
