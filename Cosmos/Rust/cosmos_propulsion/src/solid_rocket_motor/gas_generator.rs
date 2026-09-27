//! A solid rocket motor run as a gas generator: grain chamber -> outlet
//! orifice -> plenum (gas chamber) -> N pintle valves -> ambient.
//!
//! The plenum is Turns' well-stirred reactor without reaction (Turns 3e Eqs.
//! 6.28, 6.34, pp. 194-196), with wall heat loss h_w (T - T_wall):
//!   dm/dt = m_dot_12 - sum_i m_dot_i
//!   dE/dt = c_p T_up m_dot_12 - c_p T sum_i m_dot_i - h_w (T - T_wall)
//! Closing valves raises the plenum pressure; once the orifice unchokes the
//! grain-chamber pressure rises too (Sutton 9e Fig. 12-27, p. 483). Solid gas
//! generators with outlet orifices: Huzel & Huang pp. 116, 148-149.
//! Derivation note, section 7b. C++ twin: `GasGeneratorValveSystem.h`.

use cosmos_numerical::field::RealField;

use super::compressible_flow::{forward_restriction_mass_flow, ideal_thrust, restriction_mass_flow};
use super::grain::Grain;
use super::grain_chamber::GrainChamber;
use super::pintle_valve::PintleValve;
use super::runge_kutta_4::runge_kutta_4_step_stopping_at;
use super::SolidRocketMotorError;

pub const BURNED_WEB: usize = 0;
pub const CHAMBER_GAS_MASS: usize = 1;
pub const PLENUM_GAS_MASS: usize = 2;
pub const PLENUM_INTERNAL_ENERGY: usize = 3;
pub const EXPELLED_MASS: usize = 4;
pub const TOTAL_IMPULSE: usize = 5;
pub const BURNED_PROPELLANT_MASS: usize = 6;
pub const STATE_SIZE: usize = 7;

pub type State<T> = [T; STATE_SIZE];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plenum<T: RealField>
{
  volume: T,
  /// h_w in W/K; zero for an adiabatic plenum.
  wall_heat_conductance: T,
  wall_temperature: T,
}

impl<T: RealField> Plenum<T>
{
  pub fn new(volume: T, wall_heat_conductance: T, wall_temperature: T) -> Result<Self, SolidRocketMotorError>
  {
    if !(volume > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveVolume);
    }
    if !(wall_heat_conductance >= T::zero())
    {
      return Err(SolidRocketMotorError::NegativeHeatConductance);
    }
    if !(wall_temperature > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveTemperature);
    }
    Ok(Self { volume, wall_heat_conductance, wall_temperature })
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasGeneratorSample<T: RealField>
{
  pub time: T,
  pub burned_web: T,
  pub chamber_pressure: T,
  pub plenum_pressure: T,
  pub plenum_temperature: T,
  pub burning_rate: T,
  pub generation_rate: T,
  pub orifice_mass_flow: T,
  pub valve_mass_flow: T,
  pub thrust: T,
  pub expelled_mass: T,
  pub total_impulse: T,
  pub burned_propellant_mass: T,
  /// burned - dm_c - dm_plenum - expelled; zero up to round-off.
  pub mass_residual: T,
}

impl<T: RealField> GasGeneratorSample<T>
{
  pub fn as_row(&self) -> [T; 14]
  {
    [
      self.time,
      self.burned_web,
      self.chamber_pressure,
      self.plenum_pressure,
      self.plenum_temperature,
      self.burning_rate,
      self.generation_rate,
      self.orifice_mass_flow,
      self.valve_mass_flow,
      self.thrust,
      self.expelled_mass,
      self.total_impulse,
      self.burned_propellant_mass,
      self.mass_residual,
    ]
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GasGeneratorValveSystem<T: RealField, G: Grain<T>>
{
  chamber: GrainChamber<T, G>,
  orifice_area: T,
  orifice_discharge_coefficient: T,
  plenum: Plenum<T>,
  valves: Vec<PintleValve<T>>,
  ambient_pressure: T,
}

impl<T: RealField, G: Grain<T>> GasGeneratorValveSystem<T, G>
{
  pub fn new(
    chamber: GrainChamber<T, G>,
    orifice_area: T,
    orifice_discharge_coefficient: T,
    plenum: Plenum<T>,
    valves: Vec<PintleValve<T>>,
    ambient_pressure: T,
  ) -> Result<Self, SolidRocketMotorError>
  {
    if !(orifice_area > T::zero())
    {
      return Err(SolidRocketMotorError::NonPositiveArea);
    }
    if !(orifice_discharge_coefficient > T::zero())
    {
      return Err(SolidRocketMotorError::DischargeCoefficientOutOfRange);
    }
    if !(ambient_pressure >= T::zero())
    {
      return Err(SolidRocketMotorError::NegativeAmbientPressure);
    }
    Ok(Self { chamber, orifice_area, orifice_discharge_coefficient, plenum, valves, ambient_pressure })
  }

  pub fn chamber(&self) -> &GrainChamber<T, G> { &self.chamber }
  pub fn valves(&self) -> &[PintleValve<T>] { &self.valves }

  pub fn initial_state(
    &self,
    initial_chamber_pressure: T,
    initial_plenum_pressure: T,
    initial_plenum_temperature: T,
  ) -> State<T>
  {
    let gas = self.chamber.products();
    let plenum_mass = initial_plenum_pressure * self.plenum.volume
      / (gas.specific_gas_constant() * initial_plenum_temperature);
    let mut state = [T::zero(); STATE_SIZE];
    state[CHAMBER_GAS_MASS] = self.chamber.gas_mass_at(initial_chamber_pressure, T::zero());
    state[PLENUM_GAS_MASS] = plenum_mass;
    state[PLENUM_INTERNAL_ENERGY] =
      plenum_mass * gas.specific_heat_at_constant_volume() * initial_plenum_temperature;
    state
  }

  pub fn chamber_pressure(&self, state: &State<T>) -> T
  {
    self.chamber.pressure(state[CHAMBER_GAS_MASS], state[BURNED_WEB])
  }

  /// T = E / (m c_v).
  pub fn plenum_temperature(&self, state: &State<T>) -> T
  {
    state[PLENUM_INTERNAL_ENERGY]
      / (state[PLENUM_GAS_MASS] * self.chamber.products().specific_heat_at_constant_volume())
  }

  /// p = m R T / V.
  pub fn plenum_pressure(&self, state: &State<T>) -> T
  {
    state[PLENUM_GAS_MASS] * self.chamber.products().specific_gas_constant() * self.plenum_temperature(state)
      / self.plenum.volume
  }

  pub fn orifice_mass_flow(&self, state: &State<T>) -> T
  {
    restriction_mass_flow(
      self.chamber.products(),
      self.orifice_discharge_coefficient * self.orifice_area,
      self.chamber_pressure(state),
      self.chamber.products().stagnation_temperature(),
      self.plenum_pressure(state),
      self.plenum_temperature(state),
    )
  }

  pub fn valve_mass_flow(&self, valve_index: usize, time: T, pressure: T, temperature: T) -> T
  {
    let valve = &self.valves[valve_index];
    if pressure <= self.ambient_pressure
    {
      return T::zero();
    }
    forward_restriction_mass_flow(
      self.chamber.products(),
      valve.discharge_coefficient() * valve.flow_area_at(time),
      pressure,
      temperature,
      self.ambient_pressure,
    )
  }

  pub fn valve_thrust(&self, valve_index: usize, time: T, pressure: T, temperature: T) -> T
  {
    let valve = &self.valves[valve_index];
    ideal_thrust(
      self.chamber.products(),
      valve.discharge_coefficient(),
      valve.flow_area_at(time),
      valve.exit_area(),
      pressure,
      temperature,
      self.ambient_pressure,
    )
  }

  fn valve_totals(&self, time: T, pressure: T, temperature: T) -> (T, T)
  {
    let mut outflow = T::zero();
    let mut thrust = T::zero();
    for i in 0..self.valves.len()
    {
      outflow = outflow + self.valve_mass_flow(i, time, pressure, temperature);
      thrust = thrust + self.valve_thrust(i, time, pressure, temperature);
    }
    (outflow, thrust)
  }

  pub fn derivatives(&self, time: T, state: &State<T>) -> State<T>
  {
    let gas = self.chamber.products();
    let y = state[BURNED_WEB];
    let p_chamber = self.chamber_pressure(state);
    let t_plenum = self.plenum_temperature(state);
    let p_plenum = self.plenum_pressure(state);

    let r = self.chamber.burning_rate(p_chamber, y);
    let generation = self.chamber.generation_rate(r, y);
    let inflow = restriction_mass_flow(
      gas,
      self.orifice_discharge_coefficient * self.orifice_area,
      p_chamber,
      gas.stagnation_temperature(),
      p_plenum,
      t_plenum,
    );
    let (outflow, thrust) = self.valve_totals(time, p_plenum, t_plenum);

    let c_p = gas.specific_heat_at_constant_pressure();
    let upstream_temperature = if inflow >= T::zero() { gas.stagnation_temperature() } else { t_plenum };

    let mut rate = [T::zero(); STATE_SIZE];
    rate[BURNED_WEB] = r;
    rate[CHAMBER_GAS_MASS] = generation - inflow;
    rate[PLENUM_GAS_MASS] = inflow - outflow;
    rate[PLENUM_INTERNAL_ENERGY] = c_p * upstream_temperature * inflow
      - c_p * t_plenum * outflow
      - self.plenum.wall_heat_conductance * (t_plenum - self.plenum.wall_temperature);
    rate[EXPELLED_MASS] = outflow;
    rate[TOTAL_IMPULSE] = thrust;
    rate[BURNED_PROPELLANT_MASS] = generation;
    rate
  }

  pub fn sample(&self, time: T, state: &State<T>, initial: &State<T>) -> GasGeneratorSample<T>
  {
    let y = state[BURNED_WEB];
    let p_chamber = self.chamber_pressure(state);
    let t_plenum = self.plenum_temperature(state);
    let p_plenum = self.plenum_pressure(state);
    let r = self.chamber.burning_rate(p_chamber, y);
    let (outflow, thrust) = self.valve_totals(time, p_plenum, t_plenum);
    GasGeneratorSample {
      time,
      burned_web: y,
      chamber_pressure: p_chamber,
      plenum_pressure: p_plenum,
      plenum_temperature: t_plenum,
      burning_rate: r,
      generation_rate: self.chamber.generation_rate(r, y),
      orifice_mass_flow: self.orifice_mass_flow(state),
      valve_mass_flow: outflow,
      thrust,
      expelled_mass: state[EXPELLED_MASS],
      total_impulse: state[TOTAL_IMPULSE],
      burned_propellant_mass: state[BURNED_PROPELLANT_MASS],
      mass_residual: state[BURNED_PROPELLANT_MASS]
        - (state[CHAMBER_GAS_MASS] - initial[CHAMBER_GAS_MASS])
        - (state[PLENUM_GAS_MASS] - initial[PLENUM_GAS_MASS])
        - state[EXPELLED_MASS],
    }
  }

  pub fn simulate(
    &self,
    initial: &State<T>,
    step: T,
    step_count: usize,
    sample_interval: usize,
  ) -> Vec<GasGeneratorSample<T>>
  {
    assert!(step > T::zero() && sample_interval > 0);
    let mut state = *initial;
    let f = |t: T, s: &State<T>| self.derivatives(t, s);
    let web = self.chamber.grain().web();
    let mut samples = vec![self.sample(T::zero(), &state, initial)];
    for n in 1..=step_count
    {
      let t = T::from_f64((n - 1) as f64) * step;
      state = runge_kutta_4_step_stopping_at(&f, t, &state, step, BURNED_WEB, web);
      if n % sample_interval == 0 || n == step_count
      {
        samples.push(self.sample(T::from_f64(n as f64) * step, &state, initial));
      }
    }
    samples
  }
}
