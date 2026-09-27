//! Solid-propellant burning rate: Saint-Robert pressure law, two grain
//! temperature laws, and lumped Lenoir-Robillard erosive burning.
//!
//! r = a p^n (Sutton 9e Eq. 12-5, p. 446; Hill & Peterson Eq. 12.25, p. 598).
//! Exponential law: sigma_p = d ln a / dT_b constant (Sutton Eq. 12-12,
//! p. 450). Explosion-temperature law: m = c p^n / (T_e - T_0) (Williams 2e
//! Eq. 7-41, p. 250). Erosive: Sutton Eq. 12-17, p. 454, with G taken at the
//! aft end, G = rho_b A_b r / A_p, which reduces the equation to
//! r = r_0 + C r^0.8 with a unique root r >= r_0 (proof in the derivation
//! note, section 2).
//! C++ twin: `Propulsion/SolidRocketMotor/BurningRate.h`.

use cosmos_numerical::field::RealField;

use super::SolidRocketMotorError;

/// How a depends on the initial grain temperature T_b; a(T_ref) = a_ref.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemperatureSensitivity<T: RealField>
{
  /// a = a_ref exp(sigma_p (T_b - T_ref)).
  Exponential { sigma_p: T, reference_temperature: T },
  /// a = a_ref (T_e - T_ref) / (T_e - T_b).
  ExplosionTemperature { explosion_temperature: T, reference_temperature: T },
}

impl<T: RealField> TemperatureSensitivity<T>
{
  pub fn exponential(sigma_p: T, reference_temperature: T) -> Result<Self, SolidRocketMotorError>
  {
    if !(reference_temperature > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveTemperature);
    }
    Ok(Self::Exponential { sigma_p, reference_temperature })
  }

  pub fn explosion_temperature(
    explosion_temperature: T,
    reference_temperature: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(reference_temperature > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveTemperature);
    }
    if !(explosion_temperature > reference_temperature)
    {
      return Err(SolidRocketMotorError::ExplosionTemperatureNotAboveReference);
    }
    Ok(Self::ExplosionTemperature { explosion_temperature, reference_temperature })
  }

  /// a(T_b) / a(T_ref).
  pub fn coefficient_factor(&self, grain_temperature: T) -> T
  {
    match *self
    {
      Self::Exponential { sigma_p, reference_temperature } =>
      {
        (sigma_p * (grain_temperature - reference_temperature)).exponential()
      }
      Self::ExplosionTemperature { explosion_temperature, reference_temperature } =>
      {
        (explosion_temperature - reference_temperature) / (explosion_temperature - grain_temperature)
      }
    }
  }

  /// sigma_p = d ln a / dT_b at T_b.
  pub fn sigma_p(&self, grain_temperature: T) -> T
  {
    match *self
    {
      Self::Exponential { sigma_p, .. } => sigma_p,
      Self::ExplosionTemperature { explosion_temperature, .. } =>
      {
        T::one() / (explosion_temperature - grain_temperature)
      }
    }
  }
}

/// Lenoir-Robillard parameters. The exponents 0.8 and -0.2 belong to the
/// correlation (turbulent-pipe heat transfer) and are named constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosiveBurning<T: RealField>
{
  alpha: T,
  /// Sutton p. 454: "about 53" in SI.
  beta: T,
  relative_tolerance: T,
  maximum_iterations: usize,
}

impl<T: RealField> ErosiveBurning<T>
{
  pub const MASS_FLUX_EXPONENT: f64 = 0.8;
  pub const DIAMETER_EXPONENT: f64 = -0.2;
  pub const DEFAULT_RELATIVE_TOLERANCE: f64 = 1.0e-14;
  pub const DEFAULT_MAXIMUM_ITERATIONS: usize = 200;

  pub fn new(
    alpha: T,
    beta: T,
    relative_tolerance: T,
    maximum_iterations: usize,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(alpha >= T::zero()) || !(beta >= T::zero())
    {
      return Err(SolidRocketMotorError::NegativeErosiveParameter);
    }
    if !(relative_tolerance > T::zero()) || maximum_iterations == 0
    {
      return Err(SolidRocketMotorError::NonPositiveTolerance);
    }
    Ok(Self { alpha, beta, relative_tolerance, maximum_iterations })
  }

  pub fn with_default_solver(alpha: T, beta: T) -> Result<Self, SolidRocketMotorError>
  {
    Self::new(
      alpha,
      beta,
      T::from_f64(Self::DEFAULT_RELATIVE_TOLERANCE),
      Self::DEFAULT_MAXIMUM_ITERATIONS,
    )
  }

  pub fn alpha(&self) -> T { self.alpha }
  pub fn beta(&self) -> T { self.beta }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaintRobertBurningRate<T: RealField>
{
  reference_coefficient: T,
  pressure_exponent: T,
  temperature_sensitivity: TemperatureSensitivity<T>,
}

impl<T: RealField> SaintRobertBurningRate<T>
{
  /// `reference_coefficient` is a at T_ref in m/s per Pa^n; n < 1 for a
  /// stable chamber (Williams p. 250).
  pub fn new(
    reference_coefficient: T,
    pressure_exponent: T,
    temperature_sensitivity: TemperatureSensitivity<T>,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(reference_coefficient > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveBurningRateCoefficient);
    }
    if !(pressure_exponent >= T::zero())
    {
      return Err(SolidRocketMotorError::NegativePressureExponent);
    }
    Ok(Self { reference_coefficient, pressure_exponent, temperature_sensitivity })
  }

  pub fn pressure_exponent(&self) -> T { self.pressure_exponent }

  pub fn temperature_sensitivity(&self) -> &TemperatureSensitivity<T>
  {
    &self.temperature_sensitivity
  }

  pub fn coefficient(&self, grain_temperature: T) -> T
  {
    self.reference_coefficient * self.temperature_sensitivity.coefficient_factor(grain_temperature)
  }

  /// r_0 = a(T_b) p^n; zero at non-positive pressure.
  pub fn rate(&self, pressure: T, grain_temperature: T) -> T
  {
    if pressure <= T::zero()
    {
      return T::zero();
    }
    self.coefficient(grain_temperature) * pressure.power(self.pressure_exponent)
  }
}

/// Unique root r >= r_0 of r = r_0 + C r^0.8, with
/// C = alpha (rho_b A_b / A_p)^0.8 D^-0.2 exp(-beta A_p / A_b).
pub fn erosive_burning_rate<T: RealField>(
  base_rate: T,
  erosive: &ErosiveBurning<T>,
  propellant_density: T,
  burning_area: T,
  port_area: T,
  port_hydraulic_diameter: T,
) -> T
{
  let zero = T::zero();
  if base_rate <= zero || burning_area <= zero || port_area <= zero || port_hydraulic_diameter <= zero
  {
    return base_rate;
  }
  let mass_flux_exponent = T::from_f64(ErosiveBurning::<T>::MASS_FLUX_EXPONENT);
  let diameter_exponent = T::from_f64(ErosiveBurning::<T>::DIAMETER_EXPONENT);
  let coefficient = erosive.alpha
    * (propellant_density * burning_area / port_area).power(mass_flux_exponent)
    * port_hydraulic_diameter.power(diameter_exponent)
    * (-erosive.beta * port_area / burning_area).exponential();
  if coefficient <= zero
  {
    return base_rate;
  }

  let residual = |r: T| -> T { r - coefficient * r.power(mass_flux_exponent) - base_rate };
  let two = T::from_f64(2.0);
  let mut lower = base_rate;
  let mut upper = two * base_rate;
  while residual(upper) <= zero
  {
    lower = upper;
    upper = two * upper;
  }
  for _ in 0..erosive.maximum_iterations
  {
    let middle = (lower + upper) / two;
    if residual(middle) <= zero
    {
      lower = middle;
    }
    else
    {
      upper = middle;
    }
    if upper - lower <= erosive.relative_tolerance * upper
    {
      break;
    }
  }
  (lower + upper) / two
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn temperature_laws_have_their_stated_sigma_p()
  {
    let sigma = 0.002_f64;
    let reference = 294.15;
    let laws = [
      TemperatureSensitivity::exponential(sigma, reference).unwrap(),
      TemperatureSensitivity::explosion_temperature(reference + 1.0 / sigma, reference).unwrap(),
    ];
    let dt = 1.0e-3;
    for law in laws.iter()
    {
      assert_eq!(law.coefficient_factor(reference), 1.0);
      for t in [250.0, 294.15, 330.0]
      {
        let numerical = (law.coefficient_factor(t + dt).ln() - law.coefficient_factor(t - dt).ln()) / (2.0 * dt);
        assert!((numerical - law.sigma_p(t)).abs() < 1.0e-9);
      }
      assert!((law.sigma_p(reference) - sigma).abs() < 1.0e-15);
    }
  }

  #[test]
  fn erosive_root_satisfies_lenoir_robillard()
  {
    let erosive = ErosiveBurning::with_default_solver(2.0e-5_f64, 53.0).unwrap();
    let (r0, density, burning_area, port_area, diameter) = (0.0068, 1760.0, 0.314, 0.00785, 0.1);
    let r = erosive_burning_rate(r0, &erosive, density, burning_area, port_area, diameter);
    let mass_flux = density * burning_area * r / port_area;
    let right_side = r0
      + 2.0e-5 * mass_flux.powf(0.8) * diameter.powf(-0.2) * (-53.0 * r * density / mass_flux).exp();
    assert!(r > r0);
    assert!((r - right_side).abs() < 1.0e-13 * r);
  }
}
