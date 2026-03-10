#ifndef ASTRODYNAMICS_TWO_BODY_ACCELERATION_H
#define ASTRODYNAMICS_TWO_BODY_ACCELERATION_H

#include "Algebra/Modules/Vectors/Vector3.h"

#include <concepts>
#include <vector>

namespace Astrodynamics
{
namespace TwoBodyAcceleration
{

template <std::floating_point Field = double>
struct AccelerationInputs
{
  using Vector3 = Algebra::Modules::Vectors::Vector3<Field>;
  Vector3 r_;

  explicit AccelerationInputs(const Vector3& r):
    r_(r)
  {}
};

template <std::floating_point Field = double>
struct NewtonsGravitation
{
  using Vector3 = Algebra::Modules::Vectors::Vector3<Field>;
  // Mass of the central body
  Field mu_;

  explicit NewtonsGravitation(Field mu):
    mu_(mu)
  {}

  Vector3 operator()(const AccelerationInputs<Field>& inputs) const
  {
    const Field rn {inputs.r_.norm()};
    return inputs.r_ * (-mu_ / (rn * rn * rn));
  }
};

template <std::floating_point Field = double>
struct J2Perturbation
{
  using Vector3 = Algebra::Modules::Vectors::Vector3<Field>;
  Field mu_;
  Field J2_;
  Field R_;

  explicit J2Perturbation(Field mu, Field J2, Field R_):
    mu_(mu),
    J2_(J2),
    R_(R_)
  {}

  Vector3 operator()(const AccelerationInputs<Field>& inputs) const
  {
    const Field r_norm {inputs.r_.norm()};
    const Field r_squared {r_norm * r_norm};
    const Field z_squared_over_r_squared {
      inputs.r_.z() * inputs.r_.z() / r_squared};
    const Field factor {
      -1.5 * mu_ * J2_ * R_ * R_ / \
        (r_squared * r_squared * r_norm)};
    return inputs.r_ * factor * (1.0 - 5.0 * z_squared_over_r_squared);
  }

  //----------------------------------------------------------------------------
  /// Orbital mechanics helpers
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  /// Analytical J2 nodal regression (rad/s) — Curtis Eq. 4.52
  //----------------------------------------------------------------------------
  Field j2_nodal_regression(const Field a, const Field e, const Field i) const
  {
    const Field n {std::sqrt(mu_ / (a * a * a))};
    const Field p {a * (1.0 - e * e)};
    return n * \
      static_cast<Field>(-1.5) * \
      J2_ * \
      (R_ / p) * \
      (R_ / p) * \
      std::cos(i);
  }

  //----------------------------------------------------------------------------
  /// Analytical J2 perigee advance (rad/s) — Curtis Eq. 4.53
  //----------------------------------------------------------------------------
  Field j2_perigee_advance(const Field a, const Field e, const Field i) const
  {
    const Field n {std::sqrt(mu_ / (a * a * a))};
    const Field p {a * (1.0 - e * e)};
    return n * \
      static_cast<Field>(0.75) * \
      J2_ * \
      (R_ / p) * \
      (R_ / p) * \
      (5.0 * std::cos(i) * std::cos(i) - 1.0);
  }
};

template <std::floating_point Field = double>
using AccelerationFunctor = std::function<
  Algebra::Modules::Vectors::Vector3<Field>(const AccelerationInputs<Field>&)>;

template <std::floating_point Field = double>
struct TotalAcceleration
{
  using Vector3 = Algebra::Modules::Vectors::Vector3<Field>;
  std::vector<AccelerationFunctor<Field>> acceleration_functors_;

  void add(AccelerationFunctor<Field> acceleration_functor)
  {
    acceleration_functors_.push_back(acceleration_functor);
  }

  Vector3 operator()(const AccelerationInputs<Field>& inputs) const
  {
    Vector3 acceleration {};
    for (const auto& acceleration_functor : acceleration_functors_)
    {
      acceleration += acceleration_functor(inputs);
    }
    return acceleration;
  }
};

} // namespace TwoBodyAcceleration
} // namespace Astrodynamics

#endif // ASTRODYNAMICS_TWO_BODY_ACCELERATION_H