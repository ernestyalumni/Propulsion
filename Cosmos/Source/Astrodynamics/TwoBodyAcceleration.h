#ifndef ASTRODYNAMICS_TWO_BODY_ACCELERATION_H
#define ASTRODYNAMICS_TWO_BODY_ACCELERATION_H

#include "Algebra/Modules/Vectors/Vector3.h"

#include <cmath>
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
  Vector3 v_;  ///< velocity [m/s]; used by drag; defaults to zero

  /// Position-only constructor (backward compatible — v defaults to zero)
  explicit AccelerationInputs(const Vector3& r)
    : r_(r), v_(Vector3{})
  {}

  /// Full constructor with position and velocity
  AccelerationInputs(const Vector3& r, const Vector3& v)
    : r_(r), v_(v)
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

  //----------------------------------------------------------------------------
  /// Cartesian J2 acceleration (Curtis Eq. 4.51 / Bate §9.3):
  ///   a_x = factor * x * (1 - 5*z²/r²)
  ///   a_y = factor * y * (1 - 5*z²/r²)
  ///   a_z = factor * z * (3 - 5*z²/r²)   ← NOTE: coefficient is 3, not 1
  /// where factor = -1.5 * mu * J2 * R² / r^5.
  ///
  /// The z-component differs from x/y because the J2 potential is
  /// axisymmetric about the z-axis (zonal harmonic P_2(cos θ)), which
  /// introduces an extra ∂/∂z term proportional to (3z² - r²).
  //----------------------------------------------------------------------------
  Vector3 operator()(const AccelerationInputs<Field>& inputs) const
  {
    const Field r_norm {inputs.r_.norm()};
    const Field r_squared {r_norm * r_norm};
    const Field z_sq_over_r_sq {
      inputs.r_.z() * inputs.r_.z() / r_squared};
    const Field factor {
      -1.5 * mu_ * J2_ * R_ * R_ /
        (r_squared * r_squared * r_norm)};
    return Vector3{
      factor * inputs.r_.x() * (1.0 - 5.0 * z_sq_over_r_sq),
      factor * inputs.r_.y() * (1.0 - 5.0 * z_sq_over_r_sq),
      factor * inputs.r_.z() * (3.0 - 5.0 * z_sq_over_r_sq)};
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

// ---------------------------------------------------------------------------
/// Exponential atmosphere + ballistic drag (simple LEO model).
///
/// Drag acceleration:
///   a_drag = -0.5 * B * rho(r) * |v| * v
///
/// where:
///   B     = Cd * A / m   [m^2/kg]  ballistic coefficient
///   rho   = rho0 * exp(-(|r| - Re) / H)   [kg/m^3]
///   |v|   = speed [m/s]
///   v     = velocity vector [m/s]  (from AccelerationInputs::v_)
///
/// Default parameters (representative LEO satellite):
///   rho0 = 1.225  kg/m^3  (sea-level density)
///   H    = 8500.0 m       (scale height; conservative for 200-600 km)
///   B    = 2.2e-3 m^2/kg  (Cd~2.2, A/m~0.001 m^2/kg, e.g. small sat)
///
/// Assumptions:
///   - Velocity relative to atmosphere ≈ inertial velocity (no Earth rotation).
///     Error ~0.3% for LEO, acceptable for demo purposes.
///   - Exponential density model breaks down above ~1000 km; fine for LEO.
///
/// References: Curtis §10.4; Vallado §8.5; Bate §9.6.
// ---------------------------------------------------------------------------
template <std::floating_point Field = double>
struct AtmosphericDrag
{
  using Vector3 = Algebra::Modules::Vectors::Vector3<Field>;

  Field B_;     ///< Ballistic coefficient Cd*A/m [m^2/kg]
  Field Re_;    ///< Reference (equatorial) radius [m]
  Field rho0_;  ///< Sea-level density [kg/m^3]
  Field H_;     ///< Atmospheric scale height [m]

  explicit AtmosphericDrag(
      Field B,
      Field Re,
      Field rho0 = static_cast<Field>(1.225),
      Field H    = static_cast<Field>(8500.0))
    : B_(B), Re_(Re), rho0_(rho0), H_(H)
  {}

  Vector3 operator()(const AccelerationInputs<Field>& inputs) const
  {
    const Field r_norm    {inputs.r_.norm()};
    const Field altitude  {r_norm - Re_};
    const Field rho       {rho0_ * std::exp(-altitude / H_)};
    const Field v_norm    {inputs.v_.norm()};
    // a_drag = -0.5 * B * rho * |v| * v
    return inputs.v_ * static_cast<Field>(-0.5 * B_ * rho * v_norm);
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