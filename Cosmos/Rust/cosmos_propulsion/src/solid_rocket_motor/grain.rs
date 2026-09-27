//! Grain burn-back geometry as functions of the burned web y.
//!
//! V(y) = case interior + free volume - propellant volume, so dV/dy = A_b
//! by construction (solid-ballistics.tex, eq. sb-volume-rate). Grain
//! configurations: Sutton 9e Section 12.3, p. 462. For y >= web the grain is
//! burned out: A_b = 0, no slivers.
//! C++ twin: `Propulsion/SolidRocketMotor/Grain.h`.

use cosmos_numerical::field::RealField;

use super::SolidRocketMotorError;

/// What the chamber needs from a grain.
pub trait Grain<T: RealField>
{
  fn web(&self) -> T;
  fn burning_area(&self, burned_web: T) -> T;
  fn propellant_volume(&self, burned_web: T) -> T;
  fn gas_volume(&self, burned_web: T) -> T;
  /// Zero for a grain with no port (no erosive burning).
  fn port_area(&self, burned_web: T) -> T;
  fn port_hydraulic_diameter(&self, burned_web: T) -> T;
}

fn pi<T: RealField>() -> T
{
  T::from_f64(std::f64::consts::PI)
}

/// std::clamp semantics: lower if value < lower, upper if upper < value.
fn clamp<T: RealField>(value: T, lower: T, upper: T) -> T
{
  if value < lower
  {
    lower
  }
  else if upper < value
  {
    upper
  }
  else
  {
    value
  }
}

/// Internal-burning cylinder, port radius a + y, outer surface bonded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TubularGrain<T: RealField>
{
  inner_radius: T,
  outer_radius: T,
  length: T,
  ends_inhibited: bool,
  free_volume: T,
}

impl<T: RealField> TubularGrain<T>
{
  pub fn new(
    inner_radius: T,
    outer_radius: T,
    length: T,
    ends_inhibited: bool,
    free_volume: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(inner_radius > T::zero()) || !(outer_radius > inner_radius) || !(length > T::zero())
    {
      return Err(SolidRocketMotorError::InvalidGrainDimensions);
    }
    if !(free_volume >= T::zero())
    {
      return Err(SolidRocketMotorError::NegativeFreeVolume);
    }
    Ok(Self { inner_radius, outer_radius, length, ends_inhibited, free_volume })
  }
}

impl<T: RealField> Grain<T> for TubularGrain<T>
{
  fn web(&self) -> T
  {
    let radial = self.outer_radius - self.inner_radius;
    if self.ends_inhibited
    {
      radial
    }
    else
    {
      radial.minimum(self.length / T::from_f64(2.0))
    }
  }

  fn burning_area(&self, burned_web: T) -> T
  {
    if burned_web >= self.web()
    {
      return T::zero();
    }
    let y = burned_web.maximum(T::zero());
    let pi = pi::<T>();
    let port_radius = self.inner_radius + y;
    let two = T::from_f64(2.0);
    if self.ends_inhibited
    {
      return two * pi * port_radius * self.length;
    }
    two * pi * port_radius * (self.length - two * y)
      + two * pi * (self.outer_radius * self.outer_radius - port_radius * port_radius)
  }

  fn propellant_volume(&self, burned_web: T) -> T
  {
    let y = clamp(burned_web, T::zero(), self.web());
    if burned_web >= self.web()
    {
      return T::zero();
    }
    let port_radius = self.inner_radius + y;
    let annulus = pi::<T>() * (self.outer_radius * self.outer_radius - port_radius * port_radius);
    if self.ends_inhibited
    {
      annulus * self.length
    }
    else
    {
      annulus * (self.length - T::from_f64(2.0) * y)
    }
  }

  fn gas_volume(&self, burned_web: T) -> T
  {
    pi::<T>() * self.outer_radius * self.outer_radius * self.length + self.free_volume
      - self.propellant_volume(burned_web)
  }

  fn port_area(&self, burned_web: T) -> T
  {
    let y = clamp(burned_web, T::zero(), self.web());
    let port_radius = self.inner_radius + y;
    pi::<T>() * port_radius * port_radius
  }

  /// D = 4 A_p / S = 2 (a + y) (Sutton p. 454).
  fn port_hydraulic_diameter(&self, burned_web: T) -> T
  {
    let y = clamp(burned_web, T::zero(), self.web());
    T::from_f64(2.0) * (self.inner_radius + y)
  }
}

/// End-burning ("cigarette") grain: neutral burning area pi b^2, no port.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndBurningGrain<T: RealField>
{
  radius: T,
  length: T,
  free_volume: T,
}

impl<T: RealField> EndBurningGrain<T>
{
  pub fn new(radius: T, length: T, free_volume: T) -> Result<Self, SolidRocketMotorError>
  {
    if !(radius > T::zero()) || !(length > T::zero())
    {
      return Err(SolidRocketMotorError::InvalidGrainDimensions);
    }
    if !(free_volume > T::zero())
    {
      return Err(SolidRocketMotorError::NegativeFreeVolume);
    }
    Ok(Self { radius, length, free_volume })
  }
}

impl<T: RealField> Grain<T> for EndBurningGrain<T>
{
  fn web(&self) -> T
  {
    self.length
  }

  fn burning_area(&self, burned_web: T) -> T
  {
    if burned_web >= self.web()
    {
      return T::zero();
    }
    pi::<T>() * self.radius * self.radius
  }

  fn propellant_volume(&self, burned_web: T) -> T
  {
    if burned_web >= self.web()
    {
      return T::zero();
    }
    let y = burned_web.maximum(T::zero());
    pi::<T>() * self.radius * self.radius * (self.length - y)
  }

  fn gas_volume(&self, burned_web: T) -> T
  {
    pi::<T>() * self.radius * self.radius * self.length + self.free_volume
      - self.propellant_volume(burned_web)
  }

  fn port_area(&self, _burned_web: T) -> T
  {
    T::zero()
  }

  fn port_hydraulic_diameter(&self, _burned_web: T) -> T
  {
    T::zero()
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  fn expect_volume_rate_is_burning_area<G: Grain<f64>>(grain: &G)
  {
    let h = 1.0e-7;
    for i in 1..20
    {
      let y = grain.web() * (i as f64) / 20.0;
      let derivative = (grain.gas_volume(y + h) - grain.gas_volume(y - h)) / (2.0 * h);
      assert!((derivative - grain.burning_area(y)).abs() < 1.0e-6 * grain.burning_area(y));
    }
    assert_eq!(grain.burning_area(grain.web()), 0.0);
    assert_eq!(grain.propellant_volume(grain.web()), 0.0);
  }

  #[test]
  fn volume_rate_equals_burning_area_for_every_grain()
  {
    expect_volume_rate_is_burning_area(&TubularGrain::new(0.05, 0.10, 1.0, true, 0.003).unwrap());
    expect_volume_rate_is_burning_area(&TubularGrain::new(0.05, 0.10, 1.0, false, 0.003).unwrap());
    expect_volume_rate_is_burning_area(&TubularGrain::new(0.05, 0.30, 0.2, false, 0.003).unwrap());
    expect_volume_rate_is_burning_area(&EndBurningGrain::new(0.1456, 0.02, 0.0015).unwrap());
  }
}
