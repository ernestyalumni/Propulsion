//------------------------------------------------------------------------------
/// \file GrainChamber.h
/// \brief The lumped grain chamber: grain geometry, propellant, burning rate
///   and combustion products. It holds gas mass m_c and burned web y; the
///   pressure is recovered as p = m_c R T_0 / V(y).
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, section
/// 7a; notes/topics/solid-ballistics.tex. Mass generation rho_b A_b r is
/// Sutton 9e Eq. 12-1 (p. 444); the chamber mass balance is Sutton Eq. 12-3,
/// Hill & Peterson Eq. 12.28 (p. 599), Humble Eq. 6.36 (p. 337).
/// Rust twin: cosmos_propulsion::solid_rocket_motor::grain_chamber.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_GRAIN_CHAMBER_H
#define PROPULSION_SOLID_ROCKET_MOTOR_GRAIN_CHAMBER_H

#include "Propulsion/SolidRocketMotor/BurningRate.h"
#include "Propulsion/SolidRocketMotor/CombustionProducts.h"

#include <cassert>
#include <concepts>
#include <optional>

namespace Propulsion
{
namespace SolidRocketMotor
{

template <std::floating_point Field, typename Grain>
class GrainChamber
{
  public:

    GrainChamber(
      const Grain& grain,
      const Field propellant_density,
      const SaintRobertBurningRate<Field>& burning_rate_law,
      const std::optional<ErosiveBurning<Field>>& erosive_burning,
      const CombustionProducts<Field>& products,
      const Field grain_temperature
      ):
      grain_{grain},
      propellant_density_{propellant_density},
      burning_rate_law_{burning_rate_law},
      erosive_burning_{erosive_burning},
      products_{products},
      grain_temperature_{grain_temperature}
    {
      assert(propellant_density > static_cast<Field>(0));
      assert(grain_temperature > static_cast<Field>(0));
    }

    const Grain& grain() const
    {
      return grain_;
    }

    const CombustionProducts<Field>& products() const
    {
      return products_;
    }

    Field propellant_density() const
    {
      return propellant_density_;
    }

    const SaintRobertBurningRate<Field>& burning_rate_law() const
    {
      return burning_rate_law_;
    }

    Field grain_temperature() const
    {
      return grain_temperature_;
    }

    /// p = m_c R T_0 / V(y).
    Field pressure(const Field gas_mass, const Field burned_web) const
    {
      return gas_mass * products_.specific_gas_constant() *
        products_.stagnation_temperature() / grain_.gas_volume(burned_web);
    }

    /// m_c = p V(y) / (R T_0).
    Field gas_mass_at(const Field pressure, const Field burned_web) const
    {
      return pressure * grain_.gas_volume(burned_web) /
        (products_.specific_gas_constant() *
          products_.stagnation_temperature());
    }

    /// r(p, y): Saint-Robert with temperature law, plus lumped erosive burning
    /// when configured and the grain has a port. Zero after burnout.
    Field burning_rate(const Field pressure, const Field burned_web) const
    {
      const Field burning_area {grain_.burning_area(burned_web)};
      if (burning_area <= static_cast<Field>(0))
      {
        return static_cast<Field>(0);
      }
      const Field base_rate {
        burning_rate_law_.rate(pressure, grain_temperature_)};
      if (!erosive_burning_.has_value())
      {
        return base_rate;
      }
      return erosive_burning_rate(
        base_rate,
        *erosive_burning_,
        propellant_density_,
        burning_area,
        grain_.port_area(burned_web),
        grain_.port_hydraulic_diameter(burned_web));
    }

    /// rho_b A_b r, Sutton Eq. 12-1.
    Field generation_rate(const Field burning_rate, const Field burned_web)
      const
    {
      return propellant_density_ * grain_.burning_area(burned_web) *
        burning_rate;
    }

  private:

    Grain grain_;
    Field propellant_density_;
    SaintRobertBurningRate<Field> burning_rate_law_;
    std::optional<ErosiveBurning<Field>> erosive_burning_;
    CombustionProducts<Field> products_;
    Field grain_temperature_;
};

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_GRAIN_CHAMBER_H
