//------------------------------------------------------------------------------
/// \file GasGeneratorValveSystem.h
/// \brief A solid rocket motor run as a gas generator: grain chamber ->
///   outlet orifice -> plenum (gas chamber) -> N pintle valves -> ambient.
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, section
/// 7b. The plenum is Turns' well-stirred reactor without reaction (Turns 3e
/// Eqs. 6.28, 6.34, pp. 194-196), with wall heat loss h_w (T - T_wall):
///   dm/dt = m_dot_12 - sum_i m_dot_i
///   dE/dt = c_p T_up m_dot_12 - c_p T sum_i m_dot_i - h_w (T - T_wall)
/// Closing valves raises the plenum pressure; once the orifice unchokes, the
/// grain-chamber pressure rises too (Sutton 9e Fig. 12-27, p. 483). Solid
/// gas generators with outlet orifices: Huzel & Huang pp. 116, 148-149.
/// Rust twin: cosmos_propulsion::solid_rocket_motor::gas_generator.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_GAS_GENERATOR_VALVE_SYSTEM_H
#define PROPULSION_SOLID_ROCKET_MOTOR_GAS_GENERATOR_VALVE_SYSTEM_H

#include "Propulsion/SolidRocketMotor/CompressibleFlow.h"
#include "Propulsion/SolidRocketMotor/GrainChamber.h"
#include "Propulsion/SolidRocketMotor/PintleValve.h"
#include "Propulsion/SolidRocketMotor/RungeKutta4.h"

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace Propulsion
{
namespace SolidRocketMotor
{

template <std::floating_point Field = double>
struct Plenum
{
  Plenum(
    const Field volume,
    const Field wall_heat_conductance,
    const Field wall_temperature
    ):
    volume_{volume},
    wall_heat_conductance_{wall_heat_conductance},
    wall_temperature_{wall_temperature}
  {
    assert(volume > static_cast<Field>(0));
    assert(wall_heat_conductance >= static_cast<Field>(0));
    assert(wall_temperature > static_cast<Field>(0));
  }

  Field volume_;
  // h_w in W/K; zero for an adiabatic plenum.
  Field wall_heat_conductance_;
  Field wall_temperature_;
};

template <std::floating_point Field>
struct GasGeneratorSample
{
  Field time;
  Field burned_web;
  Field chamber_pressure;
  Field plenum_pressure;
  Field plenum_temperature;
  Field burning_rate;
  Field generation_rate;
  Field orifice_mass_flow;
  Field valve_mass_flow;
  Field thrust;
  Field expelled_mass;
  Field total_impulse;
  Field burned_propellant_mass;
  // burned - dm_c - dm_plenum - expelled; zero up to round-off.
  Field mass_residual;
};

template <std::floating_point Field, typename Grain>
class GasGeneratorValveSystem
{
  public:

    static constexpr std::size_t burned_web_index {0};
    static constexpr std::size_t chamber_gas_mass_index {1};
    static constexpr std::size_t plenum_gas_mass_index {2};
    static constexpr std::size_t plenum_internal_energy_index {3};
    static constexpr std::size_t expelled_mass_index {4};
    static constexpr std::size_t total_impulse_index {5};
    static constexpr std::size_t burned_propellant_mass_index {6};
    static constexpr std::size_t state_size {7};

    using State = std::array<Field, state_size>;

    GasGeneratorValveSystem(
      const GrainChamber<Field, Grain>& chamber,
      const Field orifice_area,
      const Field orifice_discharge_coefficient,
      const Plenum<Field>& plenum,
      std::vector<PintleValve<Field>> valves,
      const Field ambient_pressure
      ):
      chamber_{chamber},
      orifice_area_{orifice_area},
      orifice_discharge_coefficient_{orifice_discharge_coefficient},
      plenum_{plenum},
      valves_{std::move(valves)},
      ambient_pressure_{ambient_pressure}
    {
      assert(orifice_area > static_cast<Field>(0));
      assert(orifice_discharge_coefficient > static_cast<Field>(0));
      assert(ambient_pressure >= static_cast<Field>(0));
    }

    const GrainChamber<Field, Grain>& chamber() const
    {
      return chamber_;
    }

    const std::vector<PintleValve<Field>>& valves() const
    {
      return valves_;
    }

    State initial_state(
      const Field initial_chamber_pressure,
      const Field initial_plenum_pressure,
      const Field initial_plenum_temperature) const
    {
      const CombustionProducts<Field>& gas {chamber_.products()};
      const Field plenum_mass {
        initial_plenum_pressure * plenum_.volume_ /
          (gas.specific_gas_constant() * initial_plenum_temperature)};
      State state {};
      state[chamber_gas_mass_index] =
        chamber_.gas_mass_at(initial_chamber_pressure, static_cast<Field>(0));
      state[plenum_gas_mass_index] = plenum_mass;
      state[plenum_internal_energy_index] =
        plenum_mass * gas.specific_heat_at_constant_volume() *
          initial_plenum_temperature;
      return state;
    }

    Field chamber_pressure(const State& state) const
    {
      return chamber_.pressure(
        state[chamber_gas_mass_index],
        state[burned_web_index]);
    }

    /// T = E / (m c_v).
    Field plenum_temperature(const State& state) const
    {
      return state[plenum_internal_energy_index] /
        (state[plenum_gas_mass_index] *
          chamber_.products().specific_heat_at_constant_volume());
    }

    /// p = m R T / V = E (gamma - 1) / V.
    Field plenum_pressure(const State& state) const
    {
      return state[plenum_gas_mass_index] *
        chamber_.products().specific_gas_constant() *
        plenum_temperature(state) / plenum_.volume_;
    }

    Field orifice_mass_flow(const State& state) const
    {
      return restriction_mass_flow(
        chamber_.products(),
        orifice_discharge_coefficient_ * orifice_area_,
        chamber_pressure(state),
        chamber_.products().stagnation_temperature(),
        plenum_pressure(state),
        plenum_temperature(state));
    }

    Field valve_mass_flow(
      const std::size_t valve_index,
      const Field time,
      const Field pressure,
      const Field temperature) const
    {
      const PintleValve<Field>& valve {valves_[valve_index]};
      if (pressure <= ambient_pressure_)
      {
        return static_cast<Field>(0);
      }
      return forward_restriction_mass_flow(
        chamber_.products(),
        valve.discharge_coefficient() * valve.flow_area_at(time),
        pressure,
        temperature,
        ambient_pressure_);
    }

    Field valve_thrust(
      const std::size_t valve_index,
      const Field time,
      const Field pressure,
      const Field temperature) const
    {
      const PintleValve<Field>& valve {valves_[valve_index]};
      return ideal_thrust(
        chamber_.products(),
        valve.discharge_coefficient(),
        valve.flow_area_at(time),
        valve.exit_area(),
        pressure,
        temperature,
        ambient_pressure_);
    }

    State derivatives(const Field time, const State& state) const
    {
      const CombustionProducts<Field>& gas {chamber_.products()};
      const Field y {state[burned_web_index]};
      const Field p_chamber {chamber_pressure(state)};
      const Field t_plenum {plenum_temperature(state)};
      const Field p_plenum {plenum_pressure(state)};

      const Field r {chamber_.burning_rate(p_chamber, y)};
      const Field generation {chamber_.generation_rate(r, y)};
      const Field inflow {restriction_mass_flow(
        gas,
        orifice_discharge_coefficient_ * orifice_area_,
        p_chamber,
        gas.stagnation_temperature(),
        p_plenum,
        t_plenum)};

      Field outflow {static_cast<Field>(0)};
      Field thrust {static_cast<Field>(0)};
      for (std::size_t i {0}; i < valves_.size(); ++i)
      {
        outflow = outflow + valve_mass_flow(i, time, p_plenum, t_plenum);
        thrust = thrust + valve_thrust(i, time, p_plenum, t_plenum);
      }

      const Field c_p {gas.specific_heat_at_constant_pressure()};
      const Field upstream_temperature {
        inflow >= static_cast<Field>(0) ? gas.stagnation_temperature() :
          t_plenum};

      State rate {};
      rate[burned_web_index] = r;
      rate[chamber_gas_mass_index] = generation - inflow;
      rate[plenum_gas_mass_index] = inflow - outflow;
      rate[plenum_internal_energy_index] =
        c_p * upstream_temperature * inflow - c_p * t_plenum * outflow -
        plenum_.wall_heat_conductance_ * (t_plenum - plenum_.wall_temperature_);
      rate[expelled_mass_index] = outflow;
      rate[total_impulse_index] = thrust;
      rate[burned_propellant_mass_index] = generation;
      return rate;
    }

    GasGeneratorSample<Field> sample(
      const Field time,
      const State& state,
      const State& initial) const
    {
      const Field y {state[burned_web_index]};
      const Field p_chamber {chamber_pressure(state)};
      const Field t_plenum {plenum_temperature(state)};
      const Field p_plenum {plenum_pressure(state)};
      const Field r {chamber_.burning_rate(p_chamber, y)};
      Field outflow {static_cast<Field>(0)};
      Field thrust {static_cast<Field>(0)};
      for (std::size_t i {0}; i < valves_.size(); ++i)
      {
        outflow = outflow + valve_mass_flow(i, time, p_plenum, t_plenum);
        thrust = thrust + valve_thrust(i, time, p_plenum, t_plenum);
      }
      return GasGeneratorSample<Field>{
        time,
        y,
        p_chamber,
        p_plenum,
        t_plenum,
        r,
        chamber_.generation_rate(r, y),
        orifice_mass_flow(state),
        outflow,
        thrust,
        state[expelled_mass_index],
        state[total_impulse_index],
        state[burned_propellant_mass_index],
        state[burned_propellant_mass_index] -
          (state[chamber_gas_mass_index] - initial[chamber_gas_mass_index]) -
          (state[plenum_gas_mass_index] - initial[plenum_gas_mass_index]) -
          state[expelled_mass_index]};
    }

    std::vector<GasGeneratorSample<Field>> simulate(
      const State& initial,
      const Field step,
      const std::size_t step_count,
      const std::size_t sample_interval) const
    {
      assert(step > static_cast<Field>(0));
      assert(sample_interval > 0);
      State state {initial};
      const auto f = [this](const Field t, const State& s)
      {
        return derivatives(t, s);
      };

      std::vector<GasGeneratorSample<Field>> samples {};
      samples.push_back(sample(static_cast<Field>(0), state, initial));
      for (std::size_t n {1}; n <= step_count; ++n)
      {
        const Field t {static_cast<Field>(n - 1) * step};
        state = runge_kutta_4_step_stopping_at(
          f,
          t,
          state,
          step,
          burned_web_index,
          chamber_.grain().web());
        if (n % sample_interval == 0 || n == step_count)
        {
          samples.push_back(
            sample(static_cast<Field>(n) * step, state, initial));
        }
      }
      return samples;
    }

  private:

    GrainChamber<Field, Grain> chamber_;
    Field orifice_area_;
    Field orifice_discharge_coefficient_;
    Plenum<Field> plenum_;
    std::vector<PintleValve<Field>> valves_;
    Field ambient_pressure_;
};

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_GAS_GENERATOR_VALVE_SYSTEM_H
