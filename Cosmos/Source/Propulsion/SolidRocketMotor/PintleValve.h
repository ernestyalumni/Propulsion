//------------------------------------------------------------------------------
/// \file PintleValve.h
/// \brief A conical pintle in a throat, opened along a piecewise-linear
///   schedule of normalized stroke x in [0, 1].
///
/// Derivation: documents/derivations/SolidRocketMotorGasGenerator.md, section 6.
/// With pintle radius r_p = R_t - s tan(theta) in the throat plane and full
/// stroke s_full = R_t / tan(theta), the annular area is
/// A(x) = pi R_t^2 [1 - (1 - x)^2]. Pintle throttling: Sutton 9e p. 328;
/// hot-gas valves on a solid gas generator: Sutton Fig. 12-27, p. 483.
/// Rust twin: cosmos_propulsion::solid_rocket_motor::pintle_valve.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_PINTLE_VALVE_H
#define PROPULSION_SOLID_ROCKET_MOTOR_PINTLE_VALVE_H

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

namespace Propulsion
{
namespace SolidRocketMotor
{

/// Piecewise-linear opening x(t), held constant before the first and after
/// the last knot.
template <std::floating_point Field = double>
class OpeningSchedule
{
  public:

    struct Knot
    {
      Field time;
      Field opening;
    };

    explicit OpeningSchedule(std::vector<Knot> knots):
      knots_{std::move(knots)}
    {
      assert(!knots_.empty());
      for (std::size_t i {0}; i < knots_.size(); ++i)
      {
        assert(knots_[i].opening >= static_cast<Field>(0));
        assert(knots_[i].opening <= static_cast<Field>(1));
        if (i > 0)
        {
          assert(knots_[i].time > knots_[i - 1].time);
        }
      }
    }

    static OpeningSchedule constant(const Field opening)
    {
      return OpeningSchedule{{{static_cast<Field>(0), opening}}};
    }

    Field opening_at(const Field time) const
    {
      if (time <= knots_.front().time)
      {
        return knots_.front().opening;
      }
      if (time >= knots_.back().time)
      {
        return knots_.back().opening;
      }
      std::size_t upper {1};
      while (knots_[upper].time < time)
      {
        ++upper;
      }
      const Knot& a {knots_[upper - 1]};
      const Knot& b {knots_[upper]};
      const Field fraction {(time - a.time) / (b.time - a.time)};
      return a.opening + fraction * (b.opening - a.opening);
    }

  private:

    std::vector<Knot> knots_;
};

template <std::floating_point Field = double>
class PintleValve
{
  public:

    PintleValve(
      const Field throat_radius,
      const Field exit_area,
      const Field discharge_coefficient,
      OpeningSchedule<Field> schedule
      ):
      throat_radius_{throat_radius},
      exit_area_{exit_area},
      discharge_coefficient_{discharge_coefficient},
      schedule_{std::move(schedule)}
    {
      assert(throat_radius > static_cast<Field>(0));
      assert(exit_area >= full_open_area());
      assert(discharge_coefficient > static_cast<Field>(0));
      assert(discharge_coefficient <= static_cast<Field>(1));
    }

    Field full_open_area() const
    {
      return std::numbers::pi_v<Field> * throat_radius_ * throat_radius_;
    }

    /// A(x) = pi R_t^2 [1 - (1 - x)^2].
    Field flow_area(const Field opening) const
    {
      const Field x {
        std::clamp(opening, static_cast<Field>(0), static_cast<Field>(1))};
      const Field closed_fraction {static_cast<Field>(1) - x};
      return full_open_area() *
        (static_cast<Field>(1) - closed_fraction * closed_fraction);
    }

    Field flow_area_at(const Field time) const
    {
      return flow_area(schedule_.opening_at(time));
    }

    Field exit_area() const
    {
      return exit_area_;
    }

    Field discharge_coefficient() const
    {
      return discharge_coefficient_;
    }

    const OpeningSchedule<Field>& schedule() const
    {
      return schedule_;
    }

  private:

    Field throat_radius_;
    Field exit_area_;
    Field discharge_coefficient_;
    OpeningSchedule<Field> schedule_;
};

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_PINTLE_VALVE_H
