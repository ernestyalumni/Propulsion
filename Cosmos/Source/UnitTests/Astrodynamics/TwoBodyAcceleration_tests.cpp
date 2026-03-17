#include "Algebra/Modules/Vectors/Vector3.h"
#include "Astrodynamics/TwoBodyAcceleration.h"

#include "gtest/gtest.h"

#include <type_traits>

using Algebra::Modules::Vectors::Vector3;
using Astrodynamics::TwoBodyAcceleration::AccelerationInputs;
using Astrodynamics::TwoBodyAcceleration::NewtonsGravitation;
using Astrodynamics::TwoBodyAcceleration::J2Perturbation;
using Astrodynamics::TwoBodyAcceleration::TotalAcceleration;

namespace GoogleUnitTests
{
namespace Astrodynamics
{
namespace TwoBodyAcceleration
{

// Earth constants (approximate)
constexpr double kMu {3.986004418e14};   // m^3/s^2
constexpr double kREarth {6.371e6};      // m
constexpr double kJ2 {1.08263e-3};

// If you need it, force lookup in global namespace with
// :: prefix, e.g.
// ::Astrodynamics::TwoBodyAcceleration::NewtonsGravitation

//------------------------------------------------------------------------------
/// AccelerationInputs constructs correctly
//------------------------------------------------------------------------------
TEST(AccelerationInputsTests, ConstructsFromVector3)
{
  const Vector3<double> r {1.0, 2.0, 3.0};
  const AccelerationInputs<double> inputs {r};
  EXPECT_DOUBLE_EQ(inputs.r_.x(), 1.0);
  EXPECT_DOUBLE_EQ(inputs.r_.y(), 2.0);
  EXPECT_DOUBLE_EQ(inputs.r_.z(), 3.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TwoBodyAccelerationTests, NewtonsGravitationConstructs)
{
  constexpr bool can_construct =
    std::is_constructible_v<NewtonsGravitation<double>, double>;
  EXPECT_TRUE(can_construct);
}

//------------------------------------------------------------------------------
// Nominal: Newton only, exact doubles (r on x-axis, mu and r chosen for exact
// a).
//------------------------------------------------------------------------------
TEST(TwoBodyAccelerationTests, NewtonsGravitationOnlyExact)
{
  constexpr double mu {4.0};
  const Vector3<double> r {2.0, 0.0, 0.0};
  ::Astrodynamics::TwoBodyAcceleration::NewtonsGravitation<double> grav {mu};
  AccelerationInputs<double> inputs {r};
  const Vector3<double> a {grav(inputs)};
  // a = -mu/|r|^3 * r; |r|=2, |r|^3=8 => a = (-1, 0, 0)
  const Vector3<double> expected {-1.0, 0.0, 0.0};
  EXPECT_EQ(a, expected);
}

//------------------------------------------------------------------------------
/// NewtonsGravitation: acceleration is anti-parallel to r and magnitude = mu/r²
//------------------------------------------------------------------------------
TEST(NewtonsGravitationTests, AntiParallelAndMagnitude)
{
  const double r_mag {7.0e6};  // ~7000 km radius
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const NewtonsGravitation<double> grav {kMu};

  const Vector3<double> a {grav(inputs)};

  // Direction must be anti-parallel: a_x < 0, a_y == 0, a_z == 0
  EXPECT_LT(a.x(), 0.0);
  EXPECT_DOUBLE_EQ(a.y(), 0.0);
  EXPECT_DOUBLE_EQ(a.z(), 0.0);

  // Magnitude = mu / r²
  const double expected_mag {kMu / (r_mag * r_mag)};
  EXPECT_NEAR(a.norm(), expected_mag, 1.0e-3 * expected_mag);
}

//------------------------------------------------------------------------------
/// NewtonsGravitation: near-zero acceleration at large radius (infinity limit)
//------------------------------------------------------------------------------
TEST(NewtonsGravitationTests, NearZeroAtLargeRadius)
{
  const double r_mag {1.0e12};  // very far away
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const NewtonsGravitation<double> grav {kMu};

  const Vector3<double> a {grav(inputs)};

  EXPECT_NEAR(a.norm(), 0.0, 1.0e-9);
}

//------------------------------------------------------------------------------
/// NewtonsGravitation: centripetal check — |a| = v_circ² / r
//------------------------------------------------------------------------------
TEST(NewtonsGravitationTests, CentripetalConsistency)
{
  const double r_mag {7.0e6};
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const NewtonsGravitation<double> grav {kMu};

  const Vector3<double> a {grav(inputs)};

  // Circular orbit: v_circ = sqrt(mu / r)
  const double v_circ {std::sqrt(kMu / r_mag)};
  const double centripetal {v_circ * v_circ / r_mag};

  EXPECT_NEAR(a.norm(), centripetal, 1.0e-6 * centripetal);
}

//------------------------------------------------------------------------------
/// J2Perturbation: at equator (z=0), z-component is zero; x,y nonzero
//------------------------------------------------------------------------------
TEST(J2PerturbationTests, EquatorZComponentIsZero)
{
  const double r_mag {7.0e6};
  // Position at equator: z = 0
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const J2Perturbation<double> j2 {kMu, kJ2, kREarth};

  const Vector3<double> a {j2(inputs)};

  // z = 0 → factor * z * (3 - 0) = 0
  EXPECT_DOUBLE_EQ(a.z(), 0.0);
  // x-component is nonzero (factor * x * (1 - 0))
  EXPECT_NE(a.x(), 0.0);
  // y = 0 since r_vec.y() = 0
  EXPECT_DOUBLE_EQ(a.y(), 0.0);
}

//------------------------------------------------------------------------------
/// J2Perturbation: at north pole (r=[0,0,r]), x=0, y=0, z-component is POSITIVE.
///
/// Physical meaning: J2 > 0 encodes Earth's equatorial bulge. At the north
/// pole the oblate correction *reduces* gravitational pull relative to a
/// sphere (less mass directly overhead), so the J2 perturbation points
/// away from Earth in the +z direction.
///
/// Algebra (StormerRule.tex §5.6, eq. aJ2-compact):
///   factor = -1.5 * mu * J2 * R^2 / r^5  < 0
///   a_z    = factor * z * (3 - 5*z^2/r^2)
///          = factor * r_mag * (3 - 5)     [z = r_mag at north pole]
///          = factor * r_mag * (-2)
///          = (-)(+)(-2)  =  POSITIVE
/// Exact: a_z = 3 * mu * J2 * R^2 / r^4
//------------------------------------------------------------------------------
TEST(J2PerturbationTests, NorthPoleXYZeroZPositive)
{
  const double r_mag {7.0e6};
  const Vector3<double> r_vec {0.0, 0.0, r_mag};
  const AccelerationInputs<double> inputs {r_vec};
  const J2Perturbation<double> j2 {kMu, kJ2, kREarth};

  const Vector3<double> a {j2(inputs)};

  // x = 0, y = 0: factor * 0 * (...) = 0
  EXPECT_DOUBLE_EQ(a.x(), 0.0);
  EXPECT_DOUBLE_EQ(a.y(), 0.0);

  // a_z = factor * r * (3 - 5) > 0  since factor < 0
  EXPECT_GT(a.z(), 0.0);

  // Exact value: a_z = 3 * mu * J2 * R^2 / r^4
  const double r4 {r_mag * r_mag * r_mag * r_mag};
  const double expected_az {3.0 * kMu * kJ2 * kREarth * kREarth / r4};
  EXPECT_NEAR(a.z(), expected_az, 1.0e-9 * expected_az);
}

//------------------------------------------------------------------------------
// Nominal: Newton + J2, fictional but exact (mu=4, R=1, J2=0.25, r=(2,0,0)).
// Total acceleration = (-35/32, 0, 0) = (-1.09375, 0, 0).
//------------------------------------------------------------------------------
TEST(TwoBodyAccelerationTests, NewtonsGravitationAndJ2Exact)
{
  constexpr double mu {4.0};
  constexpr double R {1.0};
  constexpr double J2 {0.25};
  const Vector3<double> r {2.0, 0.0, 0.0};
  TotalAcceleration<double> total {};
  total.add(NewtonsGravitation<double>(mu));
  total.add(J2Perturbation<double>(mu, J2, R));
  AccelerationInputs<double> inputs {r};
  const Vector3<double> a {total(inputs)};
  // Newton: (-1,0,0). J2: factor = -3/64, a_J2 = (-3/32,0,0).
  // Sum = (-35/32, 0, 0)
  // -1.09375, exact in binary
  constexpr double expected_x {-35.0 / 32.0};
  const Vector3<double> expected {expected_x, 0.0, 0.0};
  EXPECT_EQ(a, expected);
}


} // namespace TwoBodyAcceleration
} // namespace Astrodynamics
} // namespace GoogleUnitTests