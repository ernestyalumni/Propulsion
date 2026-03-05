//------------------------------------------------------------------------------
/// \file test_DOPRI5.cpp
/// \brief Unit tests for the Dormand-Prince 5(4) stepper.
///
/// Tests mirror the Python test_CalculateNewYAndError expectations so we can
/// cross-validate the two implementations.
//------------------------------------------------------------------------------
#include "Numerical/ODE/DOPRI5.hpp"
#include "Numerical/ODE/ODEDriver.hpp"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <array>

using State1  = std::array<double, 1>;
using DOPRI5_1 = Numerical::ODE::DOPRI5<1>;

namespace {

auto example_deriv = [](double t, const State1& y) -> State1 {
  return State1{ y[0] - t*t + 1.0 };
};

auto exact = [](double t) -> double {
  return (t+1.0)*(t+1.0) - 0.5 * std::exp(t);
};

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(DOPRI5Tests)

//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithInitialConditions)
{
  DOPRI5_1 s({0.5}, 0.0);
  BOOST_TEST(s.x == 0.0);
  BOOST_TEST(s.y[0] == 0.5);
}

//------------------------------------------------------------------------------
// After one do_step from t=0 with h=0.5, yout should match exact to 5th order
BOOST_AUTO_TEST_CASE(DoStepMatchesPythonStep1)
{
  State1 y0{0.5};
  DOPRI5_1 s(y0, 0.0);
  s.dydx = {1.5};   // f(0, 0.5) = 0.5 - 0 + 1

  s.do_step(0.5, example_deriv);

  // Python test asserts yout ≈ exact(0.5) to 5 decimal places
  BOOST_TEST(s.yout[0] == exact(0.5), boost::test_tools::tolerance(1e-5));

  // k7 (FSAL) should equal f(0.5, yout)
  const State1 k7_expected = example_deriv(0.5, s.yout);
  BOOST_TEST(s.k7[0] == k7_expected[0], boost::test_tools::tolerance(1e-10));
}

//------------------------------------------------------------------------------
// Error estimate should be small (Python asserts -2.437e-5 for step 1)
BOOST_AUTO_TEST_CASE(DoStepErrorEstimateStep1)
{
  State1 y0{0.5};
  DOPRI5_1 s(y0, 0.0);
  s.dydx = {1.5};

  s.do_step(0.5, example_deriv);

  BOOST_TEST(s.yerr[0] == -2.4370659722241367e-5,
             boost::test_tools::tolerance(1e-8));
}

//------------------------------------------------------------------------------
// Adaptive step: x advances, hdid set, y ≈ exact
BOOST_AUTO_TEST_CASE(AdaptiveStepAdvancesState)
{
  DOPRI5_1 s({0.5}, 0.0, 1e-8, 1e-8);

  s.step(0.5, example_deriv);

  BOOST_TEST(s.hdid > 0.0);
  BOOST_TEST(s.x > 0.0);
  BOOST_TEST(s.y[0] == exact(s.x), boost::test_tools::tolerance(1e-6));
}

//------------------------------------------------------------------------------
// Three full adaptive steps should all match exact
BOOST_AUTO_TEST_CASE(ThreeStepsMatchExact)
{
  DOPRI5_1 s({0.5}, 0.0, 1e-9, 1e-9);

  for (int k = 0; k < 3; ++k)
    s.step(0.5, example_deriv);

  BOOST_TEST(s.y[0] == exact(s.x), boost::test_tools::tolerance(1e-7));
}

//------------------------------------------------------------------------------
// integrate_adaptive to t=3
BOOST_AUTO_TEST_CASE(IntegrateAdaptiveTo3)
{
  DOPRI5_1 s({0.5}, 0.0, 1e-10, 1e-10);

  auto n = Numerical::ODE::integrate_adaptive(s, example_deriv, 0.0, 3.0, 0.1);

  BOOST_TEST(n > 0);
  BOOST_TEST(s.x == 3.0, boost::test_tools::tolerance(1e-12));
  BOOST_TEST(s.y[0] == exact(3.0), boost::test_tools::tolerance(1e-7));
}

//------------------------------------------------------------------------------
// FSAL: after step(), dydx should equal f(x_new, y_new)
BOOST_AUTO_TEST_CASE(FSALPropertyHolds)
{
  DOPRI5_1 s({0.5}, 0.0, 1e-9, 1e-9);
  s.step(0.5, example_deriv);

  const State1 expected_dydx = example_deriv(s.x, s.y);
  BOOST_TEST(s.dydx[0] == expected_dydx[0], boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_SUITE_END()
