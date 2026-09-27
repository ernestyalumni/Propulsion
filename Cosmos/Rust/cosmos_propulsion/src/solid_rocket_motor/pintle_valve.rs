//! A conical pintle in a throat, opened along a piecewise-linear schedule of
//! normalized stroke x in [0, 1]: A(x) = pi R_t^2 [1 - (1 - x)^2].
//!
//! Derivation note, section 6. Pintle throttling: Sutton 9e p. 328; hot-gas
//! valves on a solid gas generator: Sutton Fig. 12-27, p. 483.
//! C++ twin: `Propulsion/SolidRocketMotor/PintleValve.h`.

use cosmos_numerical::field::RealField;

use super::SolidRocketMotorError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knot<T: RealField>
{
  pub time: T,
  pub opening: T,
}

/// Piecewise-linear opening x(t), held at the end values outside the knots.
#[derive(Clone, Debug, PartialEq)]
pub struct OpeningSchedule<T: RealField>
{
  knots: Vec<Knot<T>>,
}

impl<T: RealField> OpeningSchedule<T>
{
  pub fn new(knots: Vec<Knot<T>>) -> Result<Self, SolidRocketMotorError>
  {
    if knots.is_empty()
    {
      return Err(SolidRocketMotorError::EmptySchedule);
    }
    for (i, knot) in knots.iter().enumerate()
    {
      if !(knot.opening >= T::zero()) || !(knot.opening <= T::one())
      {
        return Err(SolidRocketMotorError::OpeningOutOfRange);
      }
      if i > 0 && !(knot.time > knots[i - 1].time)
      {
        return Err(SolidRocketMotorError::KnotTimesNotIncreasing);
      }
    }
    Ok(Self { knots })
  }

  pub fn constant(opening: T) -> Result<Self, SolidRocketMotorError>
  {
    Self::new(vec![Knot { time: T::zero(), opening }])
  }

  pub fn opening_at(&self, time: T) -> T
  {
    let first = self.knots[0];
    let last = self.knots[self.knots.len() - 1];
    if time <= first.time
    {
      return first.opening;
    }
    if time >= last.time
    {
      return last.opening;
    }
    let mut upper = 1;
    while self.knots[upper].time < time
    {
      upper += 1;
    }
    let a = self.knots[upper - 1];
    let b = self.knots[upper];
    let fraction = (time - a.time) / (b.time - a.time);
    a.opening + fraction * (b.opening - a.opening)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PintleValve<T: RealField>
{
  throat_radius: T,
  exit_area: T,
  discharge_coefficient: T,
  schedule: OpeningSchedule<T>,
}

impl<T: RealField> PintleValve<T>
{
  pub fn new(
    throat_radius: T,
    exit_area: T,
    discharge_coefficient: T,
    schedule: OpeningSchedule<T>,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(throat_radius > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveArea);
    }
    if !(discharge_coefficient > T::zero()) || !(discharge_coefficient <= T::one())
    {
      return Err(SolidRocketMotorError::DischargeCoefficientOutOfRange);
    }
    let valve = Self { throat_radius, exit_area, discharge_coefficient, schedule };
    if !(exit_area >= valve.full_open_area())
    {
      return Err(SolidRocketMotorError::ExitAreaBelowThroatArea);
    }
    Ok(valve)
  }

  pub fn full_open_area(&self) -> T
  {
    T::from_f64(std::f64::consts::PI) * self.throat_radius * self.throat_radius
  }

  /// A(x) = pi R_t^2 [1 - (1 - x)^2].
  pub fn flow_area(&self, opening: T) -> T
  {
    let x = if opening < T::zero()
    {
      T::zero()
    }
    else if T::one() < opening
    {
      T::one()
    }
    else
    {
      opening
    };
    let closed_fraction = T::one() - x;
    self.full_open_area() * (T::one() - closed_fraction * closed_fraction)
  }

  pub fn flow_area_at(&self, time: T) -> T
  {
    self.flow_area(self.schedule.opening_at(time))
  }

  pub fn exit_area(&self) -> T { self.exit_area }
  pub fn discharge_coefficient(&self) -> T { self.discharge_coefficient }
  pub fn schedule(&self) -> &OpeningSchedule<T> { &self.schedule }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn area_and_schedule()
  {
    let schedule = OpeningSchedule::new(vec![
      Knot { time: 0.4_f64, opening: 1.0 },
      Knot { time: 0.5, opening: 0.0 },
    ])
    .unwrap();
    let valve = PintleValve::new(0.01, 4.0e-4, 0.95, schedule).unwrap();
    assert_eq!(valve.flow_area(0.0), 0.0);
    assert!((valve.flow_area(1.0) - std::f64::consts::PI * 1.0e-4).abs() < 1.0e-18);
    assert_eq!(valve.schedule().opening_at(0.45), 0.5);
    assert_eq!(valve.schedule().opening_at(9.0), 0.0);
    assert!(OpeningSchedule::new(vec![Knot { time: 0.0_f64, opening: 1.5 }]).is_err());
  }
}
