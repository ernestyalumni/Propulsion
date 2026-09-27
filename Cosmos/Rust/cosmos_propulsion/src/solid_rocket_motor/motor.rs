//! Standalone solid rocket motor: grain chamber -> nozzle -> ambient.
//!
//! State [y, m_c, expelled, impulse, burned]:
//!   dy/dt = r(p, y),  dm_c/dt = rho_b A_b r - m_dot_nozzle(p -> p_a).
//! Derivation note, section 7a. C++ twin: `SolidRocketMotor.h`.

use cosmos_numerical::field::RealField;

use super::combustion_products::CombustionProducts;
use super::compressible_flow::{forward_restriction_mass_flow, ideal_thrust};
use super::grain::Grain;
use super::grain_chamber::GrainChamber;
use super::runge_kutta_4::runge_kutta_4_step_stopping_at;
use super::SolidRocketMotorError;

pub const BURNED_WEB: usize = 0;
pub const CHAMBER_GAS_MASS: usize = 1;
pub const EXPELLED_MASS: usize = 2;
pub const TOTAL_IMPULSE: usize = 3;
pub const BURNED_PROPELLANT_MASS: usize = 4;
pub const STATE_SIZE: usize = 5;

pub type State<T> = [T; STATE_SIZE];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorSample<T: RealField>
{
  pub time: T,
  pub burned_web: T,
  pub chamber_pressure: T,
  pub burning_rate: T,
  pub generation_rate: T,
  pub nozzle_mass_flow: T,
  pub thrust: T,
  pub expelled_mass: T,
  pub total_impulse: T,
  pub burned_propellant_mass: T,
  /// burned - (m_c - m_c(0)) - expelled; zero up to round-off.
  pub mass_residual: T,
}

impl<T: RealField> MotorSample<T>
{
  pub fn as_row(&self) -> [T; 11]
  {
    [
      self.time,
      self.burned_web,
      self.chamber_pressure,
      self.burning_rate,
      self.generation_rate,
      self.nozzle_mass_flow,
      self.thrust,
      self.expelled_mass,
      self.total_impulse,
      self.burned_propellant_mass,
      self.mass_residual,
    ]
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolidRocketMotor<T: RealField, G: Grain<T>>
{
  chamber: GrainChamber<T, G>,
  throat_area: T,
  exit_area: T,
  discharge_coefficient: T,
  ambient_pressure: T,
}

impl<T: RealField, G: Grain<T>> SolidRocketMotor<T, G>
{
  pub fn new(
    chamber: GrainChamber<T, G>,
    throat_area: T,
    exit_area: T,
    discharge_coefficient: T,
    ambient_pressure: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(throat_area > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveArea);
    }
    if !(exit_area >= throat_area)
    {
      return Err(SolidRocketMotorError::ExitAreaBelowThroatArea);
    }
    if !(discharge_coefficient > T::zero())
    {
      return Err(SolidRocketMotorError::DischargeCoefficientOutOfRange);
    }
    if !(ambient_pressure >= T::zero())
    {
      return Err(SolidRocketMotorError::NegativeAmbientPressure);
    }
    Ok(Self { chamber, throat_area, exit_area, discharge_coefficient, ambient_pressure })
  }

  pub fn chamber(&self) -> &GrainChamber<T, G> { &self.chamber }
  pub fn throat_area(&self) -> T { self.throat_area }

  fn products(&self) -> &CombustionProducts<T> { self.chamber.products() }

  pub fn initial_state(&self, initial_pressure: T) -> State<T>
  {
    let mut state = [T::zero(); STATE_SIZE];
    state[CHAMBER_GAS_MASS] = self.chamber.gas_mass_at(initial_pressure, T::zero());
    state
  }

  pub fn chamber_pressure(&self, state: &State<T>) -> T
  {
    self.chamber.pressure(state[CHAMBER_GAS_MASS], state[BURNED_WEB])
  }

  pub fn nozzle_mass_flow(&self, pressure: T) -> T
  {
    if pressure <= self.ambient_pressure
    {
      return T::zero();
    }
    forward_restriction_mass_flow(
      self.products(),
      self.discharge_coefficient * self.throat_area,
      pressure,
      self.products().stagnation_temperature(),
      self.ambient_pressure,
    )
  }

  pub fn thrust(&self, pressure: T) -> T
  {
    ideal_thrust(
      self.products(),
      self.discharge_coefficient,
      self.throat_area,
      self.exit_area,
      pressure,
      self.products().stagnation_temperature(),
      self.ambient_pressure,
    )
  }

  pub fn derivatives(&self, _time: T, state: &State<T>) -> State<T>
  {
    let y = state[BURNED_WEB];
    let p = self.chamber_pressure(state);
    let r = self.chamber.burning_rate(p, y);
    let generation = self.chamber.generation_rate(r, y);
    let outflow = self.nozzle_mass_flow(p);
    let mut rate = [T::zero(); STATE_SIZE];
    rate[BURNED_WEB] = r;
    rate[CHAMBER_GAS_MASS] = generation - outflow;
    rate[EXPELLED_MASS] = outflow;
    rate[TOTAL_IMPULSE] = self.thrust(p);
    rate[BURNED_PROPELLANT_MASS] = generation;
    rate
  }

  pub fn sample(&self, time: T, state: &State<T>, initial_gas_mass: T) -> MotorSample<T>
  {
    let y = state[BURNED_WEB];
    let p = self.chamber_pressure(state);
    let r = self.chamber.burning_rate(p, y);
    MotorSample {
      time,
      burned_web: y,
      chamber_pressure: p,
      burning_rate: r,
      generation_rate: self.chamber.generation_rate(r, y),
      nozzle_mass_flow: self.nozzle_mass_flow(p),
      thrust: self.thrust(p),
      expelled_mass: state[EXPELLED_MASS],
      total_impulse: state[TOTAL_IMPULSE],
      burned_propellant_mass: state[BURNED_PROPELLANT_MASS],
      mass_residual: state[BURNED_PROPELLANT_MASS]
        - (state[CHAMBER_GAS_MASS] - initial_gas_mass)
        - state[EXPELLED_MASS],
    }
  }

  /// Fixed-step RK4 from t = 0, landing on burnout; a sample every
  /// `sample_interval` steps plus the initial and final states.
  pub fn simulate(
    &self,
    initial_pressure: T,
    step: T,
    step_count: usize,
    sample_interval: usize,
  ) -> Vec<MotorSample<T>>
  {
    assert!(step > T::zero() && sample_interval > 0);
    let mut state = self.initial_state(initial_pressure);
    let initial_gas_mass = state[CHAMBER_GAS_MASS];
    let f = |t: T, s: &State<T>| self.derivatives(t, s);
    let web = self.chamber.grain().web();
    let mut samples = vec![self.sample(T::zero(), &state, initial_gas_mass)];
    for n in 1..=step_count
    {
      let t = T::from_f64((n - 1) as f64) * step;
      state = runge_kutta_4_step_stopping_at(&f, t, &state, step, BURNED_WEB, web);
      if n % sample_interval == 0 || n == step_count
      {
        samples.push(self.sample(T::from_f64(n as f64) * step, &state, initial_gas_mass));
      }
    }
    samples
  }
}
