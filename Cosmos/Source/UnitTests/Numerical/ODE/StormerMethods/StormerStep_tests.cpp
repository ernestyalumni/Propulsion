#include "Numerical/ODE/StormerMethods/StormerState.h"
#include "Numerical/ODE/StormerMethods/StormerStep.h"

#include "gtest/gtest.h"

#include <cmath>
#include <valarray>

using Numerical::ODE::StormerMethods::StormerState;
using Numerical::ODE::StormerMethods::StormerStep;
using std::size_t;
using std::valarray;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace StormerMethods
{

inline constexpr double epsilon {1.0e-10};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StormerStateTests, Constructs)
{
  StormerState<valarray<double>> state {
    valarray<double>{1.0},
    valarray<double>{0.0}};

  ASSERT_EQ(state.position.size(), 1);
  ASSERT_EQ(state.velocity.size(), 1);

  EXPECT_DOUBLE_EQ(state.position[0], 1.0);
  EXPECT_DOUBLE_EQ(state.velocity[0], 0.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StormerStepTests, ConstructsFromLValueDerivative)
{
  auto harmonic_acceleration = [](
    const double,
    const valarray<double>& y)
  {
    return valarray<double>{-y[0]};
  };

  StormerStep<decltype(harmonic_acceleration)> stepper {harmonic_acceleration};

  const auto acceleration =
    stepper.compute_acceleration(0.0, valarray<double>{2.0});

  EXPECT_DOUBLE_EQ(acceleration[0], -2.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StormerStepTests, ConstructsFromRValueDerivative)
{
  StormerStep stepper {
    [](const double, const valarray<double>& y)
    {
      return valarray<double>{-y[0]};
    }};

  const auto acceleration =
    stepper.compute_acceleration(0.0, valarray<double>{3.0});

  EXPECT_DOUBLE_EQ(acceleration[0], -3.0);
}

//------------------------------------------------------------------------------
/// Constant-acceleration system: y'' = a.
///
/// Exact:
///   y(H) = y_0 + v_0 H + 1/2 a H^2
///   v(H) = v_0 + a H
///
/// Störmer-Verlet is exact for constant acceleration (up to roundoff), even with
/// n_eff = 1.
//------------------------------------------------------------------------------
TEST(StormerStepTests, IntegratesConstantAccelerationExactly)
{
  constexpr double a {2.0};
  constexpr double x_0 {0.0};
  constexpr double H {0.8};

  const StormerState<valarray<double>> state {
    valarray<double>{1.0},
    valarray<double>{3.0}};

  auto constant_acceleration = [](
    const double,
    const valarray<double>&)
  {
    return valarray<double>{a};
  };

  StormerStep<decltype(constant_acceleration)> stepper {constant_acceleration};

  const valarray<double> f_0 {
    stepper.compute_acceleration(x_0, state.position)};

  const auto result = stepper.step(x_0, state, f_0, H, 1);

  const double y_exact {1.0 + 3.0 * H + 0.5 * a * H * H};
  const double v_exact {3.0 + a * H};

  EXPECT_NEAR(result.position[0], y_exact, epsilon);
  EXPECT_NEAR(result.velocity[0], v_exact, epsilon);
}

//------------------------------------------------------------------------------
/// Harmonic oscillator: y'' = -y, y(0) = 1, y'(0)=0.
///
/// Exact:
///   y(x) = cos(x), v(x) = -sin(x)
//------------------------------------------------------------------------------
TEST(StormerStepTests, IntegratesHarmonicOscillatorToExpectedAccuracy)
{
  constexpr double x_0 {0.0};
  constexpr double H {0.5};

  const StormerState<valarray<double>> state {
    valarray<double>{1.0},
    valarray<double>{0.0}};

  auto harmonic_acceleration = [](
    const double,
    const valarray<double>& y)
  {
    return valarray<double>{-y[0]};
  };

  StormerStep<decltype(harmonic_acceleration)> stepper {harmonic_acceleration};

  const valarray<double> f_0 {
    stepper.compute_acceleration(x_0, state.position)};

  const auto result = stepper.step(x_0, state, f_0, H, 128);

  const double y_exact {std::cos(H)};
  const double v_exact {-std::sin(H)};

  EXPECT_NEAR(result.position[0], y_exact, 1.0e-6);
  EXPECT_NEAR(result.velocity[0], v_exact, 1.0e-6);
}

//------------------------------------------------------------------------------
/// Convergence check (global O(h_eff^2)):
///
/// For fixed interval H, error should reduce by ~4 when n_eff doubles.
//------------------------------------------------------------------------------
TEST(StormerStepTests, DemonstratesSecondOrderConvergence)
{
  constexpr double x_0 {0.0};
  constexpr double H {0.5};

  const StormerState<valarray<double>> state {
    valarray<double>{1.0},
    valarray<double>{0.0}};

  auto harmonic_acceleration = [](
    const double,
    const valarray<double>& y)
  {
    return valarray<double>{-y[0]};
  };

  StormerStep<decltype(harmonic_acceleration)> stepper {harmonic_acceleration};

  const valarray<double> f_0 {
    stepper.compute_acceleration(x_0, state.position)};

  const auto coarse = stepper.step(x_0, state, f_0, H, 16);
  const auto fine = stepper.step(x_0, state, f_0, H, 32);

  const double y_exact {std::cos(H)};

  const double e_coarse {std::abs(coarse.position[0] - y_exact)};
  const double e_fine {std::abs(fine.position[0] - y_exact)};

  ASSERT_GT(e_fine, 0.0);

  const double ratio {e_coarse / e_fine};

  // Second-order expectation is ratio ~ 4; allow slack.
  EXPECT_GT(ratio, 3.0);
}

} // namespace StormerMethods
} // namespace ODE
} // namespace Numerical
} // namespace GoogleUnitTests
