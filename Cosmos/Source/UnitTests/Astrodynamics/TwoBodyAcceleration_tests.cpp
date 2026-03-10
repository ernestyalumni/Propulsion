#include "Astrodynamics/TwoBodyAcceleration.h"

#include "gtest/gtest.h"

#include <cmath>

using Astrodynamics::TwoBodyAcceleration::AccelerationInputs;
using Astrodynamics::TwoBodyAcceleration::J2Perturbation;
using Astrodynamics::TwoBodyAcceleration::NewtonsGravitation;
using Algebra::Modules::Vectors::Vector3;

namespace GoogleUnitTests
{
namespace Astrodynamics
{
namespace TwoBodyAcceleration
{

// Earth-like constants for testing
constexpr double kMu {3.986004418e14};   // m^3/s^2
constexpr double kREarth {6.371e6};      // m
constexpr double kJ2 {1.08263e-3};

//------------------------------------------------------------------------------
/// AccelerationInputs: constructs and stores r correctly
//------------------------------------------------------------------------------
TEST(AccelerationInputsTests, ConstructsCorrectly)
{
  const Vector3<double> r {1.0, 2.0, 3.0};
  const AccelerationInputs<double> inputs {r};
  EXPECT_DOUBLE_EQ(inputs.r_.x(), 1.0);
  EXPECT_DOUBLE_EQ(inputs.r_.y(), 2.0);
  EXPECT_DOUBLE_EQ(inputs.r_.z(), 3.0);
}

//------------------------------------------------------------------------------
/// NewtonsGravitation: acceleration is anti-parallel to r; magnitude = mu/r²
//------------------------------------------------------------------------------
TEST(NewtonsGravitationTests, AntiParallelAndMagnitude)
{
  const double r_mag {7.0e6};
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const NewtonsGravitation<double> grav {kMu};

  const Vector3<double> a {grav(inputs)};

  // Direction: anti-parallel to r_vec → a_x < 0, a_y = 0, a_z = 0
  EXPECT_LT(a.x(), 0.0);
  EXPECT_DOUBLE_EQ(a.y(), 0.0);
  EXPECT_DOUBLE_EQ(a.z(), 0.0);

  // Magnitude: mu / r²
  const double expected_mag {kMu / (r_mag * r_mag)};
  EXPECT_NEAR(a.norm(), expected_mag, 1.0e-6 * expected_mag);
}

//------------------------------------------------------------------------------
/// NewtonsGravitation: near-zero acceleration at very large radius (infinity limit)
//------------------------------------------------------------------------------
TEST(NewtonsGravitationTests, NearZeroAtLargeRadius)
{
  const double r_mag {1.0e12};  // ~6700 AU — effectively infinity
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const NewtonsGravitation<double> grav {kMu};

  const Vector3<double> a {grav(inputs)};

  EXPECT_NEAR(a.norm(), 0.0, 1.0e-9);
}

//------------------------------------------------------------------------------
/// NewtonsGravitation: circular orbit centripetal check — |a| = v_circ² / r
//------------------------------------------------------------------------------
TEST(NewtonsGravitationTests, CentripetalConsistency)
{
  const double r_mag {7.0e6};
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const NewtonsGravitation<double> grav {kMu};

  const Vector3<double> a {grav(inputs)};

  // Circular orbit: v_circ = sqrt(mu / r), centripetal = v_circ² / r = mu / r²
  const double v_circ {std::sqrt(kMu / r_mag)};
  const double centripetal {v_circ * v_circ / r_mag};

  EXPECT_NEAR(a.norm(), centripetal, 1.0e-6 * centripetal);
}

//------------------------------------------------------------------------------
/// J2Perturbation: at equator (z=0), z-component is zero; x-component is nonzero
//------------------------------------------------------------------------------
TEST(J2PerturbationTests, EquatorZComponentIsZero)
{
  const double r_mag {7.0e6};
  // Position on equator along x-axis
  const Vector3<double> r_vec {r_mag, 0.0, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const J2Perturbation<double> j2 {kMu, kJ2, kREarth};

  const Vector3<double> a {j2(inputs)};

  // z = 0 → a_z = factor * 0 * (...) = 0
  EXPECT_DOUBLE_EQ(a.z(), 0.0);
  // factor < 0, r_vec.x() > 0, (1 - 5*0) = 1 → a_x < 0 (nonzero)
  EXPECT_NE(a.x(), 0.0);
  // r_vec.y() = 0 → a_y = 0
  EXPECT_DOUBLE_EQ(a.y(), 0.0);
}

//------------------------------------------------------------------------------
/// J2Perturbation: at equator with nonzero x and y, both x and y components
/// are nonzero while z remains zero
//------------------------------------------------------------------------------
TEST(J2PerturbationTests, EquatorXYComponentsNonzeroWhenBothPresent)
{
  const double r_mag {7.0e6};
  const double c {r_mag / std::sqrt(2.0)};
  // Position on equatorial plane with x and y both nonzero
  const Vector3<double> r_vec {c, c, 0.0};
  const AccelerationInputs<double> inputs {r_vec};
  const J2Perturbation<double> j2 {kMu, kJ2, kREarth};

  const Vector3<double> a {j2(inputs)};

  EXPECT_NE(a.x(), 0.0);
  EXPECT_NE(a.y(), 0.0);
  EXPECT_DOUBLE_EQ(a.z(), 0.0);
}

//------------------------------------------------------------------------------
/// J2Perturbation: at north pole (r=[0,0,r]), x=0, y=0; z-component is positive
///
/// At the north pole (z = r, x = y = 0):
///   factor = -1.5 * mu * J2 * R² / r^5  < 0
///   a_z = factor * r * (3 - 5*1) = factor * r * (-2)  > 0
///
/// The J2 perturbation at the pole has a positive z-component because the
/// oblate Earth's equatorial bulge slightly reduces the centripetal pull
/// in the z direction (analogous to how the pole is "closer" to the center
/// but the effective gravity gradient from the J2 potential adds outward
/// along z).
//------------------------------------------------------------------------------
TEST(J2PerturbationTests, NorthPoleXYZeroZPositive)
{
  const double r_mag {7.0e6};
  const Vector3<double> r_vec {0.0, 0.0, r_mag};
  const AccelerationInputs<double> inputs {r_vec};
  const J2Perturbation<double> j2 {kMu, kJ2, kREarth};

  const Vector3<double> a {j2(inputs)};

  // x = 0: factor * 0 * (...) = 0
  EXPECT_DOUBLE_EQ(a.x(), 0.0);
  // y = 0: factor * 0 * (...) = 0
  EXPECT_DOUBLE_EQ(a.y(), 0.0);
  // z: factor < 0, z = r > 0, (3 - 5) = -2 → a_z = (-)(r)(-2) > 0
  EXPECT_GT(a.z(), 0.0);
}

} // namespace TwoBodyAcceleration
} // namespace Astrodynamics
} // namespace GoogleUnitTests
