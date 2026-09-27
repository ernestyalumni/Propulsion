//------------------------------------------------------------------------------
/// \file CombustionProducts.h
/// \brief The combustion gas as a calorically perfect gas with fixed
///   stagnation temperature, molar mass and ratio of specific heats.
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, section 1.
/// R = R_u / molar mass (Turns 3e Eq. 2.3, p. 13); flow function Gamma (Hill &
/// Peterson 2e Eq. 3.14, p. 71); c* = sqrt(R T_0) / Gamma (Sutton 9e Eq. 3-32,
/// p. 63). T_0 does not depend on chamber pressure (Hill & Peterson p. 599).
/// Rust twin: cosmos_propulsion::solid_rocket_motor::combustion_products.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_COMBUSTION_PRODUCTS_H
#define PROPULSION_SOLID_ROCKET_MOTOR_COMBUSTION_PRODUCTS_H

#include <cassert>
#include <cmath>
#include <concepts>

namespace Propulsion
{
namespace SolidRocketMotor
{

/// CODATA 2018 molar gas constant, J / (mol K). A named default, injected
/// wherever it is used.
inline constexpr double universal_gas_constant_si {8.314462618};

/// Gamma(gamma) = sqrt(gamma) (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1))),
/// Hill & Peterson Eq. 3.14. Choked mass flux is Gamma p_0 / sqrt(R T_0).
template <std::floating_point Field = double>
Field flow_function(const Field heat_capacity_ratio)
{
  assert(heat_capacity_ratio > static_cast<Field>(1));
  const Field gamma {heat_capacity_ratio};
  const Field one {static_cast<Field>(1)};
  const Field two {static_cast<Field>(2)};
  return std::sqrt(gamma) *
    std::pow(two / (gamma + one), (gamma + one) / (two * (gamma - one)));
}

/// Critical pressure ratio p_t / p_1 = (2 / (gamma + 1))^(gamma / (gamma - 1)),
/// Sutton Eq. 3-20.
template <std::floating_point Field = double>
Field critical_pressure_ratio(const Field heat_capacity_ratio)
{
  assert(heat_capacity_ratio > static_cast<Field>(1));
  const Field gamma {heat_capacity_ratio};
  const Field one {static_cast<Field>(1)};
  const Field two {static_cast<Field>(2)};
  return std::pow(two / (gamma + one), gamma / (gamma - one));
}

template <std::floating_point Field = double>
class CombustionProducts
{
  public:

    CombustionProducts(
      const Field stagnation_temperature,
      const Field molar_mass,
      const Field heat_capacity_ratio,
      const Field universal_gas_constant =
        static_cast<Field>(universal_gas_constant_si)
      ):
      stagnation_temperature_{stagnation_temperature},
      molar_mass_{molar_mass},
      heat_capacity_ratio_{heat_capacity_ratio},
      universal_gas_constant_{universal_gas_constant}
    {
      assert(stagnation_temperature > static_cast<Field>(0));
      assert(molar_mass > static_cast<Field>(0));
      assert(heat_capacity_ratio > static_cast<Field>(1));
      assert(universal_gas_constant > static_cast<Field>(0));
    }

    /// Invert c* = sqrt(R_u T_0 / M) / Gamma for the molar mass, for sources
    /// that quote c* and T_0 (Huzel & Huang p. 116).
    static CombustionProducts from_characteristic_velocity(
      const Field characteristic_velocity,
      const Field stagnation_temperature,
      const Field heat_capacity_ratio,
      const Field universal_gas_constant =
        static_cast<Field>(universal_gas_constant_si))
    {
      assert(characteristic_velocity > static_cast<Field>(0));
      const Field gamma_c_star {
        flow_function(heat_capacity_ratio) * characteristic_velocity};
      return CombustionProducts{
        stagnation_temperature,
        universal_gas_constant * stagnation_temperature /
          (gamma_c_star * gamma_c_star),
        heat_capacity_ratio,
        universal_gas_constant};
    }

    Field stagnation_temperature() const
    {
      return stagnation_temperature_;
    }

    Field molar_mass() const
    {
      return molar_mass_;
    }

    Field heat_capacity_ratio() const
    {
      return heat_capacity_ratio_;
    }

    /// R = R_u / M, Turns Eq. 2.3.
    Field specific_gas_constant() const
    {
      return universal_gas_constant_ / molar_mass_;
    }

    /// c_p = gamma R / (gamma - 1).
    Field specific_heat_at_constant_pressure() const
    {
      return heat_capacity_ratio_ * specific_gas_constant() /
        (heat_capacity_ratio_ - static_cast<Field>(1));
    }

    /// c_v = R / (gamma - 1).
    Field specific_heat_at_constant_volume() const
    {
      return specific_gas_constant() /
        (heat_capacity_ratio_ - static_cast<Field>(1));
    }

    Field flow_function_value() const
    {
      return flow_function(heat_capacity_ratio_);
    }

    /// c* = sqrt(R T_0) / Gamma, Sutton Eq. 3-32.
    Field characteristic_velocity() const
    {
      return std::sqrt(specific_gas_constant() * stagnation_temperature_) /
        flow_function_value();
    }

    /// Gas density at pressure p and the stagnation temperature.
    Field density_at(const Field pressure) const
    {
      return pressure / (specific_gas_constant() * stagnation_temperature_);
    }

  private:

    Field stagnation_temperature_;
    Field molar_mass_;
    Field heat_capacity_ratio_;
    Field universal_gas_constant_;
};

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_COMBUSTION_PRODUCTS_H
