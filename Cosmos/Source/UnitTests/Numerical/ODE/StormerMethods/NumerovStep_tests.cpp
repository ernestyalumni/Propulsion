#include "Numerical/ODE/StormerMethods/NumerovState.h"
#include "Numerical/ODE/StormerMethods/NumerovStep.h"
#include "Numerical/ODE/StormerMethods/StormerStep.h"

#include "gtest/gtest.h"

#include <cmath>
#include <valarray>

using Numerical::ODE::StormerMethods::NumerovState;
using Numerical::ODE::StormerMethods::NumerovStep;
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Harmonic oscillator: y'' = -omega^2 * y
/// Exact: y(t) = cos(omega*t),  y'(t) = -omega*sin(omega*t)
static constexpr double omega {1.0};

auto harmonic_accel = [](const double, const valarray<double>& y)
{
  return valarray<double>{-omega * omega * y[0]};
};

/// Constant-acceleration: y'' = a  (exact for any polynomial integrator)
static constexpr double a_const {2.0};

auto const_accel = [](const double, const valarray<double>&)
{
  return valarray<double>{a_const};
};

// ---------------------------------------------------------------------------
// NumerovState construction
// ---------------------------------------------------------------------------
TEST(NumerovStateTests, Constructs)
{
  NumerovState<valarray<double>> state {
    valarray<double>{1.0},  // position
    valarray<double>{0.0},  // velocity
    valarray<double>{0.9},  // y_prev
    valarray<double>{-0.9}  // f_prev
  };

  EXPECT_DOUBLE_EQ(state.position[0], 1.0);
  EXPECT_DOUBLE_EQ(state.velocity[0], 0.0);
  EXPECT_DOUBLE_EQ(state.y_prev[0],   0.9);
  EXPECT_DOUBLE_EQ(state.f_prev[0],  -0.9);
}

// ---------------------------------------------------------------------------
// NumerovStep construction
// ---------------------------------------------------------------------------
TEST(NumerovStepTests, ConstructsFromLValue)
{
  NumerovStep<decltype(harmonic_accel)> stepper {harmonic_accel};
  const auto f = stepper.compute_acceleration(0.0, valarray<double>{2.0});
  EXPECT_DOUBLE_EQ(f[0], -2.0);
}

TEST(NumerovStepTests, ConstructsFromRValue)
{
  NumerovStep stepper {
    [](const double, const valarray<double>& y)
    { return valarray<double>{-y[0]}; }
  };
  const auto f = stepper.compute_acceleration(0.0, valarray<double>{3.0});
  EXPECT_DOUBLE_EQ(f[0], -3.0);
}

// ---------------------------------------------------------------------------
// Constant acceleration — Numerov is EXACT for quadratic solutions
//
// y'' = a, y(0)=y0, y'(0)=v0
// Exact: y(t) = y0 + v0*t + a/2 * t^2
//
// Since the exact solution is a polynomial of degree 2, all higher-order
// derivatives vanish. Both the explicit Störmer and Numerov predict and
// correct to machine precision for such problems.
// ---------------------------------------------------------------------------
TEST(NumerovStepTests, StartupFromConstantAcceleration)
{
  constexpr double x_0 {0.0};
  constexpr double h   {0.5};
  const valarray<double> y_0 {1.0};
  const valarray<double> v_0 {3.0};

  NumerovStep stepper {const_accel};
  auto state = stepper.startup(x_0, y_0, v_0, h, 128);

  const double y_1_exact {1.0 + 3.0 * h + 0.5 * a_const * h * h};
  EXPECT_NEAR(state.position[0], y_1_exact, 1.0e-10);
}

TEST(NumerovStepTests, IntegratesConstantAccelerationExactly)
{
  // y'' = 2, y(0)=1, y'(0)=3
  // Exact: y(t) = 1 + 3t + t^2
  constexpr double x_0 {0.0};
  constexpr double h   {0.4};
  const valarray<double> y_0 {1.0};
  const valarray<double> v_0 {3.0};

  NumerovStep stepper {const_accel};
  auto state = stepper.startup(x_0, y_0, v_0, h, 256);

  // f_1 = a_const  (constant)
  const valarray<double> f_1 {stepper.compute_acceleration(x_0 + h, state.position)};

  // One more step: from step 1 to step 2
  auto state2 = stepper.step(x_0 + h, state, f_1, h);

  const double t2 {2.0 * h};
  const double y_2_exact {1.0 + 3.0 * t2 + t2 * t2};

  // Numerov is exact for quadratic y — limited only by rounding
  EXPECT_NEAR(state2.position[0], y_2_exact, 1.0e-10);
}

// ---------------------------------------------------------------------------
// Harmonic oscillator: accuracy and order tests
//
// y'' = -y, y(0) = 1, y'(0) = 0
// Exact: y(t) = cos(t), y'(t) = -sin(t)
// ---------------------------------------------------------------------------

/// Use startup + one step, measure position error at t = H.
static double harmonic_position_error_at(
  const double H,
  const size_t n_steps,      // number of Numerov steps
  const size_t n_sub_start = 512)
{
  const double h {H / static_cast<double>(n_steps)};
  const valarray<double> y_0 {1.0};
  const valarray<double> v_0 {0.0};

  NumerovStep stepper {harmonic_accel};
  auto state = stepper.startup(0.0, y_0, v_0, h, n_sub_start);

  // f_1 needed for subsequent steps
  valarray<double> f_curr {
    stepper.compute_acceleration(h, state.position)
  };

  // March n_steps - 1 more steps (startup gave step 1)
  double x = h;
  for (size_t i = 1; i < n_steps; ++i)
  {
    state  = stepper.step(x, state, f_curr, h);
    x     += h;
    f_curr = stepper.compute_acceleration(x, state.position);
  }

  const double y_exact {std::cos(H)};
  return std::abs(state.position[0] - y_exact);
}

TEST(NumerovStepTests, HarmonicOscillatorAccuracy)
{
  // At H = 1.0 with 16 steps (h = 1/16), expect well below 1e-6
  const double err = harmonic_position_error_at(1.0, 16, 512);
  EXPECT_LT(err, 1.0e-6);
}

/// Order-4 convergence: halving h should reduce error by factor ~16
TEST(NumerovStepTests, DemonstratesFourthOrderConvergence)
{
  // Use a large number of startup sub-steps so startup error << Numerov error
  constexpr double H {1.0};
  const double err_coarse = harmonic_position_error_at(H, 8,  1024);
  const double err_fine   = harmonic_position_error_at(H, 16, 1024);

  ASSERT_GT(err_fine, 0.0);
  const double ratio {err_coarse / err_fine};

  // Fourth-order: ratio ~ 2^4 = 16. Allow slack [8, 32].
  EXPECT_GT(ratio, 8.0);
  EXPECT_LT(ratio, 32.0);
}

/// Numerov must be strictly better than explicit Störmer at the same h.
/// Compare errors for the harmonic oscillator at the same step count.
TEST(NumerovStepTests, StrictlyBetterThanExplicitStormer)
{
  constexpr double H       {1.0};
  constexpr size_t n_steps {8};
  const double     h       {H / static_cast<double>(n_steps)};

  // --- Numerov error ---
  const double err_numerov = harmonic_position_error_at(H, n_steps, 512);

  // --- Explicit Störmer error (StormerStep with n_eff=1 = one true step) ---
  StormerStep stormer {harmonic_accel};
  const valarray<double> y_0 {1.0};
  const valarray<double> v_0 {0.0};

  StormerState<valarray<double>> stor_state {y_0, v_0};
  valarray<double> f_0 {stormer.compute_acceleration(0.0, y_0)};
  double x = 0.0;

  for (size_t i = 0; i < n_steps; ++i)
  {
    stor_state = stormer.step(x, stor_state, f_0, h, 1);
    x         += h;
    f_0        = stormer.compute_acceleration(x, stor_state.position);
  }
  const double err_stormer {std::abs(stor_state.position[0] - std::cos(H))};

  // Numerov should be significantly more accurate
  EXPECT_LT(err_numerov, err_stormer / 4.0);
}

// ---------------------------------------------------------------------------
// History sliding: after each step, y_prev and f_prev must be updated
// correctly (internal consistency).
// ---------------------------------------------------------------------------
TEST(NumerovStepTests, HistorySlidesCorrectly)
{
  constexpr double x_0 {0.0};
  constexpr double h   {0.1};

  const valarray<double> y_0 {1.0};
  const valarray<double> v_0 {0.0};

  NumerovStep stepper {harmonic_accel};
  auto state = stepper.startup(x_0, y_0, v_0, h, 256);

  // After startup: state.y_prev should be close to y_0 = 1.0
  // (not equal due to startup stepping, but close)
  EXPECT_NEAR(state.y_prev[0], 1.0, 1.0e-4);

  // After one step, the new state.y_prev should equal the previous position
  const valarray<double> pos_before_step {state.position};
  valarray<double> f_curr {
    stepper.compute_acceleration(h, state.position)
  };
  auto state2 = stepper.step(h, state, f_curr, h);

  EXPECT_DOUBLE_EQ(state2.y_prev[0], pos_before_step[0]);
}

} // namespace StormerMethods
} // namespace ODE
} // namespace Numerical
} // namespace GoogleUnitTests
