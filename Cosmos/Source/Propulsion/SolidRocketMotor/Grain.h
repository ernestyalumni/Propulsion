//------------------------------------------------------------------------------
/// \file Grain.h
/// \brief Grain burn-back geometry as functions of the burned web y: burning
///   area A_b(y), gas volume V(y), port area and hydraulic diameter.
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, section 3.
/// The gas volume is computed as case interior + free volume - propellant
/// volume, so dV/dy = A_b holds by construction (solid-ballistics.tex,
/// eq. sb-volume-rate). Grain configurations: Sutton 9e Section 12.3, p. 462.
/// For y >= web the grain is burned out: A_b = 0 and no slivers remain.
/// Rust twin: cosmos_propulsion::solid_rocket_motor::grain.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_GRAIN_H
#define PROPULSION_SOLID_ROCKET_MOTOR_GRAIN_H

#include <algorithm>
#include <cassert>
#include <concepts>
#include <numbers>

namespace Propulsion
{
namespace SolidRocketMotor
{

/// Internal-burning cylinder (tube). Port radius a + y; the outer surface is
/// bonded to the case. Ends are either inhibited or burning.
template <std::floating_point Field = double>
class TubularGrain
{
  public:

    TubularGrain(
      const Field inner_radius,
      const Field outer_radius,
      const Field length,
      const bool ends_inhibited,
      const Field free_volume
      ):
      inner_radius_{inner_radius},
      outer_radius_{outer_radius},
      length_{length},
      ends_inhibited_{ends_inhibited},
      free_volume_{free_volume}
    {
      assert(inner_radius > static_cast<Field>(0));
      assert(outer_radius > inner_radius);
      assert(length > static_cast<Field>(0));
      assert(free_volume >= static_cast<Field>(0));
    }

    /// w = b - a (inhibited ends) or min(b - a, L / 2) (burning ends).
    Field web() const
    {
      const Field radial {outer_radius_ - inner_radius_};
      return ends_inhibited_ ? radial :
        std::min(radial, length_ / static_cast<Field>(2));
    }

    Field burning_area(const Field burned_web) const
    {
      if (burned_web >= web())
      {
        return static_cast<Field>(0);
      }
      const Field y {std::max(burned_web, static_cast<Field>(0))};
      const Field pi {std::numbers::pi_v<Field>};
      const Field port_radius {inner_radius_ + y};
      const Field two {static_cast<Field>(2)};
      if (ends_inhibited_)
      {
        return two * pi * port_radius * length_;
      }
      return two * pi * port_radius * (length_ - two * y) +
        two * pi * (outer_radius_ * outer_radius_ - port_radius * port_radius);
    }

    Field propellant_volume(const Field burned_web) const
    {
      const Field y {std::clamp(burned_web, static_cast<Field>(0), web())};
      if (burned_web >= web())
      {
        return static_cast<Field>(0);
      }
      const Field pi {std::numbers::pi_v<Field>};
      const Field port_radius {inner_radius_ + y};
      const Field annulus {
        pi * (outer_radius_ * outer_radius_ - port_radius * port_radius)};
      return ends_inhibited_ ? annulus * length_ :
        annulus * (length_ - static_cast<Field>(2) * y);
    }

    /// Case interior (pi b^2 L) + free volume - propellant volume.
    Field gas_volume(const Field burned_web) const
    {
      const Field pi {std::numbers::pi_v<Field>};
      return pi * outer_radius_ * outer_radius_ * length_ + free_volume_ -
        propellant_volume(burned_web);
    }

    Field port_area(const Field burned_web) const
    {
      const Field y {std::clamp(burned_web, static_cast<Field>(0), web())};
      const Field port_radius {inner_radius_ + y};
      return std::numbers::pi_v<Field> * port_radius * port_radius;
    }

    /// D = 4 A_p / S = 2 (a + y) for a circular port (Sutton p. 454).
    Field port_hydraulic_diameter(const Field burned_web) const
    {
      const Field y {std::clamp(burned_web, static_cast<Field>(0), web())};
      return static_cast<Field>(2) * (inner_radius_ + y);
    }

  private:

    Field inner_radius_;
    Field outer_radius_;
    Field length_;
    bool ends_inhibited_;
    Field free_volume_;
};

/// End-burning ("cigarette") grain: burns on one face only, neutral burning
/// area pi b^2. The usual long-duration gas-generator grain. No port, so no
/// erosive burning.
template <std::floating_point Field = double>
class EndBurningGrain
{
  public:

    EndBurningGrain(
      const Field radius,
      const Field length,
      const Field free_volume
      ):
      radius_{radius},
      length_{length},
      free_volume_{free_volume}
    {
      assert(radius > static_cast<Field>(0));
      assert(length > static_cast<Field>(0));
      assert(free_volume > static_cast<Field>(0));
    }

    Field web() const
    {
      return length_;
    }

    Field burning_area(const Field burned_web) const
    {
      if (burned_web >= web())
      {
        return static_cast<Field>(0);
      }
      return std::numbers::pi_v<Field> * radius_ * radius_;
    }

    Field propellant_volume(const Field burned_web) const
    {
      if (burned_web >= web())
      {
        return static_cast<Field>(0);
      }
      const Field y {std::max(burned_web, static_cast<Field>(0))};
      return std::numbers::pi_v<Field> * radius_ * radius_ * (length_ - y);
    }

    Field gas_volume(const Field burned_web) const
    {
      return std::numbers::pi_v<Field> * radius_ * radius_ * length_ +
        free_volume_ - propellant_volume(burned_web);
    }

    Field port_area(const Field) const
    {
      return static_cast<Field>(0);
    }

    Field port_hydraulic_diameter(const Field) const
    {
      return static_cast<Field>(0);
    }

  private:

    Field radius_;
    Field length_;
    Field free_volume_;
};

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_GRAIN_H
