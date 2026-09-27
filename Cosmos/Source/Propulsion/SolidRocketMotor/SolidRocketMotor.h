//------------------------------------------------------------------------------
/// \file SolidRocketMotor.h
/// \brief Standalone solid rocket motor: grain chamber -> nozzle -> ambient.
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, section
/// 7a. State: burned web y, chamber gas mass m_c, and quadratures of expelled
/// mass, total impulse and burned propellant mass.
///   dy/dt   = r(p, y)
///   dm_c/dt = rho_b A_b r - m_dot_nozzle(p -> p_a)
/// Rust twin: cosmos_propulsion::solid_rocket_motor::motor.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_SOLID_ROCKET_MOTOR_H
#define PROPULSION_SOLID_ROCKET_MOTOR_SOLID_ROCKET_MOTOR_H

#include "Propulsion/SolidRocketMotor/CompressibleFlow.h"
#include "Propulsion/SolidRocketMotor/GrainChamber.h"
#include "Propulsion/SolidRocketMotor/RungeKutta4.h"

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <vector>

namespace Propulsion
{
namespace SolidRocketMotor
{

template <std::floating_point Field>
struct MotorSample
{
  Field time;
  Field burned_web;
  Field chamber_pressure;
  Field burning_rate;
  Field generation_rate;
  Field nozzle_mass_flow;
  Field thrust;
  Field expelled_mass;
  Field total_impulse;
  Field burned_propellant_mass;
  // burned - (m_c - m_c(0)) - expelled; zero up to round-off.
  Field mass_residual;
};

template <std::floating_point Field, typename Grain>
class SolidRocketMotor
{
  public:

    static constexpr std::size_t burned_web_index {0};
    static constexpr std::size_t chamber_gas_mass_index {1};
    static constexpr std::size_t expelled_mass_index {2};
    static constexpr std::size_t total_impulse_index {3};
    static constexpr std::size_t burned_propellant_mass_index {4};
    static constexpr std::size_t state_size {5};

    using State = std::array<Field, state_size>;

    SolidRocketMotor(
      const GrainChamber<Field, Grain>& chamber,
      const Field throat_area,
      const Field exit_area,
      const Field discharge_coefficient,
      const Field ambient_pressure
      ):
      chamber_{chamber},
      throat_area_{throat_area},
      exit_area_{exit_area},
      discharge_coefficient_{discharge_coefficient},
      ambient_pressure_{ambient_pressure}
    {
      assert(throat_area > static_cast<Field>(0));
      assert(exit_area >= throat_area);
      assert(discharge_coefficient > static_cast<Field>(0));
      assert(ambient_pressure >= static_cast<Field>(0));
    }

    const GrainChamber<Field, Grain>& chamber() const
    {
      return chamber_;
    }

    Field throat_area() const
    {
      return throat_area_;
    }

    /// The motor is lit at t = 0 with chamber pressure initial_pressure.
    State initial_state(const Field initial_pressure) const
    {
      State state {};
      state[chamber_gas_mass_index] =
        chamber_.gas_mass_at(initial_pressure, static_cast<Field>(0));
      return state;
    }

    Field chamber_pressure(const State& state) const
    {
      return chamber_.pressure(
        state[chamber_gas_mass_index],
        state[burned_web_index]);
    }

    Field nozzle_mass_flow(const Field pressure) const
    {
      if (pressure <= ambient_pressure_)
      {
        return static_cast<Field>(0);
      }
      return forward_restriction_mass_flow(
        chamber_.products(),
        discharge_coefficient_ * throat_area_,
        pressure,
        chamber_.products().stagnation_temperature(),
        ambient_pressure_);
    }

    Field thrust(const Field pressure) const
    {
      return ideal_thrust(
        chamber_.products(),
        discharge_coefficient_,
        throat_area_,
        exit_area_,
        pressure,
        chamber_.products().stagnation_temperature(),
        ambient_pressure_);
    }

    State derivatives(const Field, const State& state) const
    {
      const Field y {state[burned_web_index]};
      const Field p {chamber_pressure(state)};
      const Field r {chamber_.burning_rate(p, y)};
      const Field generation {chamber_.generation_rate(r, y)};
      const Field outflow {nozzle_mass_flow(p)};

      State rate {};
      rate[burned_web_index] = r;
      rate[chamber_gas_mass_index] = generation - outflow;
      rate[expelled_mass_index] = outflow;
      rate[total_impulse_index] = thrust(p);
      rate[burned_propellant_mass_index] = generation;
      return rate;
    }

    MotorSample<Field> sample(
      const Field time,
      const State& state,
      const Field initial_gas_mass) const
    {
      const Field y {state[burned_web_index]};
      const Field p {chamber_pressure(state)};
      const Field r {chamber_.burning_rate(p, y)};
      return MotorSample<Field>{
        time,
        y,
        p,
        r,
        chamber_.generation_rate(r, y),
        nozzle_mass_flow(p),
        thrust(p),
        state[expelled_mass_index],
        state[total_impulse_index],
        state[burned_propellant_mass_index],
        state[burned_propellant_mass_index] -
          (state[chamber_gas_mass_index] - initial_gas_mass) -
          state[expelled_mass_index]};
    }

    /// Fixed-step RK4 from t = 0 for step_count steps; a sample every
    /// sample_interval steps, plus the initial and final states.
    std::vector<MotorSample<Field>> simulate(
      const Field initial_pressure,
      const Field step,
      const std::size_t step_count,
      const std::size_t sample_interval) const
    {
      assert(step > static_cast<Field>(0));
      assert(sample_interval > 0);
      State state {initial_state(initial_pressure)};
      const Field initial_gas_mass {state[chamber_gas_mass_index]};
      const auto f = [this](const Field t, const State& s)
      {
        return derivatives(t, s);
      };

      std::vector<MotorSample<Field>> samples {};
      samples.push_back(sample(static_cast<Field>(0), state, initial_gas_mass));
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
            sample(static_cast<Field>(n) * step, state, initial_gas_mass));
        }
      }
      return samples;
    }

  private:

    GrainChamber<Field, Grain> chamber_;
    Field throat_area_;
    Field exit_area_;
    Field discharge_coefficient_;
    Field ambient_pressure_;
};

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_SOLID_ROCKET_MOTOR_H
