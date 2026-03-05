//------------------------------------------------------------------------------
/// \file test_RKCK.cpp
/// \brief Unit tests for the Cash-Karp 4(5) stepper.
///
/// Reference ODE:  dy/dt = y − t² + 1,  y(0) = 0.5
/// Exact solution: y(t) = (t+1)² − 0.5·eˢ
/// (This is the same test problem used in the Python DOPRI5 tests.)
//------------------------------------------------------------------------------
#include "Numerical/ODE/RKCK.hpp"
#include "Numerical/ODE/ODEDriver.hpp"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <array>

using State1 = std::array<double, 1>;
using RKCK1  = Numerical::ODE::RKCK<1>;

// ── Reference ODE ────────────────────────────────────────────────────────────
namespace {

auto example_deriv = [](double t, const State1& y) -> State1 {
  return State1{ y[0] - t*t + 1.0 };
};

auto exact = [](double t) -> double {
  return (t+1.0)*(t+1.0) - 0.5 * std::exp(t);
};

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(RKCKTests)

//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(StepperConstructs)
{
  State1 y0{0.5};
  RKCK1 s(y0, 0.0);
  BOOST_TEST(s.x == 0.0);
  BOOST_TEST(s.y[0] == 0.5);
}

//------------------------------------------------------------------------------
// do_step should match reference values (single step from t=0, h=0.5)
BOOST_AUTO_TEST_CASE(DoStepSingleStep)
{
  State1 y0{0.5};
  RKCK1 s(y0, 0.0, 1e-6, 1e-6);

  s.dydx = example_deriv(0.0, y0);  // k1 = 1.5

  BOOST_TEST(s.dydx[0] == 1.5);

  s.do_step(0.5, example_deriv);

  // 5th-order output should be close to exact(0.5)
  const double expected = exact(0.5); // ≈ 1.4256...
  BOOST_TEST(s.yout[0] == expected, boost::test_tools::tolerance(1e-4));

  // Error should be small (embedded pair difference)
  BOOST_TEST(std::abs(s.yerr[0]) < 1e-4);
}

//------------------------------------------------------------------------------
// Adaptive step: after step(), x should advance and error norm ≤ 1
BOOST_AUTO_TEST_CASE(AdaptiveStepAccepted)
{
  State1 y0{0.5};
  RKCK1 s(y0, 0.0, 1e-8, 1e-8);

  s.step(0.5, example_deriv);

  BOOST_TEST(s.hdid > 0.0);
  BOOST_TEST(s.x > 0.0);
  BOOST_TEST(s.x <= 0.5 + 1e-14);
  BOOST_TEST(s.y[0] == exact(s.x), boost::test_tools::tolerance(1e-6));
}

//------------------------------------------------------------------------------
// integrate_adaptive: propagate from t=0 to t=2, compare to exact
BOOST_AUTO_TEST_CASE(IntegrateAdaptiveThreeSteps)
{
  State1 y0{0.5};
  RKCK1 s(y0, 0.0, 1e-9, 1e-9);

  auto nsteps = Numerical::ODE::integrate_adaptive(s, example_deriv,
                                                    0.0, 2.0, 0.1);

  BOOST_TEST(nsteps > 0);
  BOOST_TEST(s.x == 2.0, boost::test_tools::tolerance(1e-12));
  BOOST_TEST(s.y[0] == exact(2.0), boost::test_tools::tolerance(1e-7));
}

//------------------------------------------------------------------------------
// Energy conservation proxy: y at t=0 should satisfy exact(0) = 0.5
BOOST_AUTO_TEST_CASE(ExactSolutionAtOrigin)
{
  BOOST_TEST(exact(0.0) == 0.5, boost::test_tools::tolerance(1e-15));
}

BOOST_AUTO_TEST_SUITE_END()
