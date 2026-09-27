//------------------------------------------------------------------------------
/// \file BurningRate.h
/// \brief Solid-propellant burning rate: the Saint-Robert (Vieille) pressure
///   law, two grain-temperature laws, and lumped Lenoir-Robillard erosive
///   burning.
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, section 2.
/// r = a p^n (Sutton 9e Eq. 12-5, p. 446; Hill & Peterson Eq. 12.25, p. 598).
/// Exponential law: sigma_p = d ln a / dT_b constant (Sutton Eq. 12-12, p. 450).
/// Explosion-temperature law: m = c p^n / (T_e - T_0) (Williams 2e Eq. 7-41,
/// p. 250). Erosive burning: Sutton Eq. 12-17, p. 454, with G at the aft end.
/// Rust twin: cosmos_propulsion::solid_rocket_motor::burning_rate.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_BURNING_RATE_H
#define PROPULSION_SOLID_ROCKET_MOTOR_BURNING_RATE_H

#include <cassert>
#include <cmath>
#include <concepts>

namespace Propulsion
{
namespace SolidRocketMotor
{

/// How the burning-rate coefficient a depends on the initial grain
/// temperature T_b. Both are normalized so a(T_reference) = a_reference.
template <std::floating_point Field = double>
struct TemperatureSensitivity
{
  enum class Law
  {
    // a = a_ref exp(sigma_p (T_b - T_ref)), constant sigma_p (Sutton 12-12).
    exponential,
    // a = a_ref (T_e - T_ref) / (T_e - T_b), Williams 7-41.
    explosion_temperature
  };

  static TemperatureSensitivity exponential(
    const Field sigma_p,
    const Field reference_temperature)
  {
    assert(reference_temperature > static_cast<Field>(0));
    return TemperatureSensitivity{
      Law::exponential,
      sigma_p,
      static_cast<Field>(0),
      reference_temperature};
  }

  static TemperatureSensitivity explosion_temperature(
    const Field explosion_temperature,
    const Field reference_temperature)
  {
    assert(reference_temperature > static_cast<Field>(0));
    assert(explosion_temperature > reference_temperature);
    return TemperatureSensitivity{
      Law::explosion_temperature,
      static_cast<Field>(0),
      explosion_temperature,
      reference_temperature};
  }

  /// a(T_b) / a(T_ref).
  Field coefficient_factor(const Field grain_temperature) const
  {
    if (law_ == Law::exponential)
    {
      return std::exp(sigma_p_ * (grain_temperature - reference_temperature_));
    }
    assert(grain_temperature < explosion_temperature_);
    return (explosion_temperature_ - reference_temperature_) /
      (explosion_temperature_ - grain_temperature);
  }

  /// sigma_p = d ln a / dT_b at T_b.
  Field sigma_p(const Field grain_temperature) const
  {
    if (law_ == Law::exponential)
    {
      return sigma_p_;
    }
    return static_cast<Field>(1) / (explosion_temperature_ - grain_temperature);
  }

  Law law_;
  Field sigma_p_;
  Field explosion_temperature_;
  Field reference_temperature_;
};

/// Lenoir-Robillard erosive augmentation, r = r_0 + alpha G^0.8 D^-0.2
/// exp(-beta r rho_b / G), Sutton Eq. 12-17. The exponents 0.8 and -0.2 are
/// part of the correlation (turbulent-pipe heat transfer), named here.
template <std::floating_point Field = double>
struct ErosiveBurning
{
  static constexpr Field mass_flux_exponent {static_cast<Field>(0.8)};
  static constexpr Field diameter_exponent {static_cast<Field>(-0.2)};

  ErosiveBurning(
    const Field alpha,
    const Field beta,
    const Field relative_tolerance = static_cast<Field>(1.0e-14),
    const int maximum_iterations = 200
    ):
    alpha_{alpha},
    beta_{beta},
    relative_tolerance_{relative_tolerance},
    maximum_iterations_{maximum_iterations}
  {
    assert(alpha >= static_cast<Field>(0));
    assert(beta >= static_cast<Field>(0));
    assert(relative_tolerance > static_cast<Field>(0));
    assert(maximum_iterations > 0);
  }

  Field alpha_;
  // Sutton p. 454: "about 53" in SI units (r m/s, G kg/m^2-s).
  Field beta_;
  Field relative_tolerance_;
  int maximum_iterations_;
};

template <std::floating_point Field = double>
class SaintRobertBurningRate
{
  public:

    /// \param reference_coefficient a at T_ref, in m/s per Pa^n.
    /// \param pressure_exponent n; n < 1 for a stable chamber (Williams p. 250).
    SaintRobertBurningRate(
      const Field reference_coefficient,
      const Field pressure_exponent,
      const TemperatureSensitivity<Field>& temperature_sensitivity
      ):
      reference_coefficient_{reference_coefficient},
      pressure_exponent_{pressure_exponent},
      temperature_sensitivity_{temperature_sensitivity}
    {
      assert(reference_coefficient > static_cast<Field>(0));
      assert(pressure_exponent >= static_cast<Field>(0));
    }

    Field pressure_exponent() const
    {
      return pressure_exponent_;
    }

    const TemperatureSensitivity<Field>& temperature_sensitivity() const
    {
      return temperature_sensitivity_;
    }

    /// a(T_b), m/s per Pa^n.
    Field coefficient(const Field grain_temperature) const
    {
      return reference_coefficient_ *
        temperature_sensitivity_.coefficient_factor(grain_temperature);
    }

    /// r_0 = a(T_b) p^n, m/s. Zero at non-positive pressure.
    Field rate(const Field pressure, const Field grain_temperature) const
    {
      if (pressure <= static_cast<Field>(0))
      {
        return static_cast<Field>(0);
      }
      return coefficient(grain_temperature) *
        std::pow(pressure, pressure_exponent_);
    }

  private:

    Field reference_coefficient_;
    Field pressure_exponent_;
    TemperatureSensitivity<Field> temperature_sensitivity_;
};

/// Solve r = r_0 + C r^0.8 for the unique root r >= r_0 (proof in the
/// derivation note, section 2), where with G = rho_b A_b r / A_p at the aft
/// end, C = alpha (rho_b A_b / A_p)^0.8 D^-0.2 exp(-beta A_p / A_b).
/// Bracket by doubling, then bisect to the relative tolerance.
template <std::floating_point Field = double>
Field erosive_burning_rate(
  const Field base_rate,
  const ErosiveBurning<Field>& erosive,
  const Field propellant_density,
  const Field burning_area,
  const Field port_area,
  const Field port_hydraulic_diameter)
{
  if (base_rate <= static_cast<Field>(0) ||
    burning_area <= static_cast<Field>(0) ||
    port_area <= static_cast<Field>(0) ||
    port_hydraulic_diameter <= static_cast<Field>(0))
  {
    return base_rate;
  }

  const Field coefficient {
    erosive.alpha_ *
      std::pow(
        propellant_density * burning_area / port_area,
        ErosiveBurning<Field>::mass_flux_exponent) *
      std::pow(
        port_hydraulic_diameter,
        ErosiveBurning<Field>::diameter_exponent) *
      std::exp(-erosive.beta_ * port_area / burning_area)};

  if (coefficient <= static_cast<Field>(0))
  {
    return base_rate;
  }

  const auto residual = [&](const Field r) -> Field
  {
    return r - coefficient * std::pow(r, ErosiveBurning<Field>::mass_flux_exponent)
      - base_rate;
  };

  Field lower {base_rate};
  Field upper {static_cast<Field>(2) * base_rate};
  while (residual(upper) <= static_cast<Field>(0))
  {
    lower = upper;
    upper = static_cast<Field>(2) * upper;
  }

  for (int iteration {0}; iteration < erosive.maximum_iterations_; ++iteration)
  {
    const Field middle {(lower + upper) / static_cast<Field>(2)};
    if (residual(middle) <= static_cast<Field>(0))
    {
      lower = middle;
    }
    else
    {
      upper = middle;
    }
    if (upper - lower <= erosive.relative_tolerance_ * upper)
    {
      break;
    }
  }
  return (lower + upper) / static_cast<Field>(2);
}

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_BURNING_RATE_H
