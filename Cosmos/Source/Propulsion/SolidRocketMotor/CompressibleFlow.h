//------------------------------------------------------------------------------
/// \file CompressibleFlow.h
/// \brief Isentropic flow through a restriction (choked, subsonic, reversed),
///   the supersonic exit Mach number for an area ratio, and ideal thrust.
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, sections
/// 4 and 5. Choked flow: Sutton 9e Eq. 3-24 (p. 59) = Hill & Peterson Eq. 3.14
/// (p. 71). Subsonic flow: Sutton Eq. 3-25 solved for the flux. Thrust
/// coefficient: Sutton Eq. 3-30 (p. 62). Exit velocity: Sutton Eq. 3-16.
/// Rust twin: cosmos_propulsion::solid_rocket_motor::compressible_flow.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_COMPRESSIBLE_FLOW_H
#define PROPULSION_SOLID_ROCKET_MOTOR_COMPRESSIBLE_FLOW_H

#include "Propulsion/SolidRocketMotor/CombustionProducts.h"

#include <cassert>
#include <cmath>
#include <concepts>

namespace Propulsion
{
namespace SolidRocketMotor
{

/// Mass flow from stagnation (p_u, T_u) to static p_d through C_d A, for
/// p_d <= p_u. Choked when p_d / p_u <= critical ratio.
template <std::floating_point Field = double>
Field forward_restriction_mass_flow(
  const CombustionProducts<Field>& products,
  const Field effective_area,
  const Field upstream_pressure,
  const Field upstream_temperature,
  const Field downstream_pressure)
{
  if (effective_area <= static_cast<Field>(0) ||
    upstream_pressure <= static_cast<Field>(0))
  {
    return static_cast<Field>(0);
  }
  const Field gamma {products.heat_capacity_ratio()};
  const Field gas_constant {products.specific_gas_constant()};
  const Field ratio {downstream_pressure / upstream_pressure};
  if (ratio <= critical_pressure_ratio(gamma))
  {
    return effective_area * upstream_pressure * flow_function(gamma) /
      std::sqrt(gas_constant * upstream_temperature);
  }
  const Field one {static_cast<Field>(1)};
  const Field two {static_cast<Field>(2)};
  const Field bracket {
    std::pow(ratio, two / gamma) - std::pow(ratio, (gamma + one) / gamma)};
  const Field positive_bracket {
    bracket > static_cast<Field>(0) ? bracket : static_cast<Field>(0)};
  return effective_area * upstream_pressure * std::sqrt(
    two * gamma / ((gamma - one) * gas_constant * upstream_temperature) *
      positive_bracket);
}

/// Signed mass flow from side a to side b. Reversed flow uses side b's
/// temperature as its upstream stagnation temperature.
template <std::floating_point Field = double>
Field restriction_mass_flow(
  const CombustionProducts<Field>& products,
  const Field effective_area,
  const Field pressure_a,
  const Field temperature_a,
  const Field pressure_b,
  const Field temperature_b)
{
  if (pressure_a >= pressure_b)
  {
    return forward_restriction_mass_flow(
      products,
      effective_area,
      pressure_a,
      temperature_a,
      pressure_b);
  }
  return -forward_restriction_mass_flow(
    products,
    effective_area,
    pressure_b,
    temperature_b,
    pressure_a);
}

/// A / A* = (1 / M) [ (2 / (gamma + 1)) (1 + (gamma - 1) M^2 / 2) ]^((gamma +
/// 1) / (2 (gamma - 1))).
template <std::floating_point Field = double>
Field area_ratio_at_mach(const Field mach, const Field heat_capacity_ratio)
{
  const Field gamma {heat_capacity_ratio};
  const Field one {static_cast<Field>(1)};
  const Field two {static_cast<Field>(2)};
  return (one / mach) * std::pow(
    (two / (gamma + one)) * (one + (gamma - one) / two * mach * mach),
    (gamma + one) / (two * (gamma - one)));
}

/// The supersonic root M >= 1 of A / A* = expansion_ratio. The area ratio is
/// increasing in M on the supersonic branch, so bracket by doubling and bisect.
template <std::floating_point Field = double>
Field supersonic_exit_mach(
  const Field expansion_ratio,
  const Field heat_capacity_ratio,
  const Field relative_tolerance = static_cast<Field>(1.0e-15),
  const int maximum_iterations = 200)
{
  assert(expansion_ratio >= static_cast<Field>(1));
  Field lower {static_cast<Field>(1)};
  if (expansion_ratio <= lower)
  {
    return lower;
  }
  Field upper {static_cast<Field>(2)};
  while (area_ratio_at_mach(upper, heat_capacity_ratio) < expansion_ratio)
  {
    lower = upper;
    upper = static_cast<Field>(2) * upper;
  }
  for (int iteration {0}; iteration < maximum_iterations; ++iteration)
  {
    const Field middle {(lower + upper) / static_cast<Field>(2)};
    if (area_ratio_at_mach(middle, heat_capacity_ratio) < expansion_ratio)
    {
      lower = middle;
    }
    else
    {
      upper = middle;
    }
    if (upper - lower <= relative_tolerance * upper)
    {
      break;
    }
  }
  return (lower + upper) / static_cast<Field>(2);
}

/// p / p_0 = (1 + (gamma - 1) M^2 / 2)^(-gamma / (gamma - 1)).
template <std::floating_point Field = double>
Field static_to_stagnation_pressure_ratio(
  const Field mach,
  const Field heat_capacity_ratio)
{
  const Field gamma {heat_capacity_ratio};
  const Field one {static_cast<Field>(1)};
  const Field two {static_cast<Field>(2)};
  return std::pow(
    one + (gamma - one) / two * mach * mach,
    -gamma / (gamma - one));
}

/// Ideal thrust C_F p_c C_d A_t (choked) or m_dot v_e with p_e = p_a (not
/// choked). Separation of an over-expanded jet is not modeled. Zero when the
/// chamber is at or below ambient.
template <std::floating_point Field = double>
Field ideal_thrust(
  const CombustionProducts<Field>& products,
  const Field discharge_coefficient,
  const Field throat_area,
  const Field exit_area,
  const Field chamber_pressure,
  const Field chamber_temperature,
  const Field ambient_pressure)
{
  if (throat_area <= static_cast<Field>(0) ||
    chamber_pressure <= ambient_pressure)
  {
    return static_cast<Field>(0);
  }
  const Field gamma {products.heat_capacity_ratio()};
  const Field one {static_cast<Field>(1)};
  const Field two {static_cast<Field>(2)};
  const Field ambient_ratio {ambient_pressure / chamber_pressure};
  if (ambient_ratio <= critical_pressure_ratio(gamma))
  {
    const Field expansion_ratio {
      exit_area > throat_area ? exit_area / throat_area : one};
    const Field exit_ratio {static_to_stagnation_pressure_ratio(
      supersonic_exit_mach(expansion_ratio, gamma),
      gamma)};
    const Field momentum {std::sqrt(
      two * gamma * gamma / (gamma - one) *
        std::pow(two / (gamma + one), (gamma + one) / (gamma - one)) *
        (one - std::pow(exit_ratio, (gamma - one) / gamma)))};
    const Field thrust_coefficient {
      momentum + (exit_ratio - ambient_ratio) * expansion_ratio};
    return thrust_coefficient * chamber_pressure * discharge_coefficient *
      throat_area;
  }
  const Field mass_flow {forward_restriction_mass_flow(
    products,
    discharge_coefficient * throat_area,
    chamber_pressure,
    chamber_temperature,
    ambient_pressure)};
  const Field exit_velocity {std::sqrt(
    two * gamma / (gamma - one) * products.specific_gas_constant() *
      chamber_temperature *
      (one - std::pow(ambient_ratio, (gamma - one) / gamma)))};
  return mass_flow * exit_velocity;
}

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_COMPRESSIBLE_FLOW_H
