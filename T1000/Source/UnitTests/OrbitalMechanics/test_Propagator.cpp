//------------------------------------------------------------------------------
/// \file test_Propagator.cpp
/// \brief Integration tests: propagate orbits and verify conserved quantities.
///
/// The key thing being tested here is that the stepper + ODE driver + physics
/// all work together, and that conserved quantities (energy, |h|) stay constant.
//------------------------------------------------------------------------------
#include "OrbitalMechanics/Propagator.hpp"
#include "OrbitalMechanics/OrbitalElements.hpp"
#include "OrbitalMechanics/TwoBody.hpp"
#include "OrbitalMechanics/Constants.hpp"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <algorithm>

using namespace OrbitalMechanics;
using namespace OrbitalMechanics::Constants;

BOOST_AUTO_TEST_SUITE(PropagatorTests)

//------------------------------------------------------------------------------
// Circular orbit: after one period, state should return close to initial.
BOOST_AUTO_TEST_CASE(CircularOrbitReturnsToStart)
{
  const double alt = 400.0e3;
  const double r   = R_EARTH + alt;
  const double vc  = circular_speed(r);
  const double T   = orbital_period(r);

  StateVector6 y0{r, 0.0, 0.0, 0.0, vc, 0.0};

  auto result = propagate(y0, 0.0, T,
                          MU_EARTH,
                          1.0e-8, 1.0e-9,
                          T / 100.0,
                          false);  // only need final state

  const StateVector6& yf = result.states.back();

  // Position returns within ~1 m (atol=1e-8 → ~mm drift over one orbit is OK).
  // Tolerance is relative to r for x, and absolute for y (which starts near 0).
  BOOST_TEST(yf[0] == y0[0], boost::test_tools::tolerance(2e-7));   // ~1m / 6.78e6 m
  BOOST_TEST(std::abs(yf[1]) < 0.1);  // y starts at 0; drift < 10 cm is fine
  BOOST_TEST(yf[2] == y0[2], boost::test_tools::tolerance(1e-12));

  // Speed is much better conserved than position (no phase drift in |v|).
  BOOST_TEST(speed(yf) == vc, boost::test_tools::tolerance(1e-8));
}

//------------------------------------------------------------------------------
// Energy conservation over one ISS orbit period (relative drift < 1e-9)
BOOST_AUTO_TEST_CASE(EnergyConservedOverOneOrbit)
{
  const double r  = R_EARTH + 400.0e3;
  const double vc = circular_speed(r);
  const double T  = orbital_period(r);

  StateVector6 y0{r, 0.0, 0.0, 0.0, vc, 0.0};

  auto result = propagate(y0, 0.0, T,
                          MU_EARTH, 1.0e-10, 1.0e-11,
                          T / 200.0, true);

  const double e0 = result.energies.front();
  const double ef = result.energies.back();

  // Relative energy drift < 1 ppb
  BOOST_TEST(std::abs((ef - e0) / e0) < 1.0e-9);
}

//------------------------------------------------------------------------------
// Angular momentum conservation (should be even better than energy)
BOOST_AUTO_TEST_CASE(AngularMomentumConserved)
{
  const double r  = R_EARTH + 600.0e3;
  const double vc = circular_speed(r);
  const double T  = orbital_period(r);

  StateVector6 y0{r, 0.0, 0.0, 0.0, vc, 0.0};

  auto result = propagate(y0, 0.0, T,
                          MU_EARTH, 1.0e-10, 1.0e-11,
                          T / 200.0, true);

  const double h0  = result.hmags.front();
  const double hmax = *std::max_element(result.hmags.begin(), result.hmags.end());
  const double hmin = *std::min_element(result.hmags.begin(), result.hmags.end());

  BOOST_TEST((hmax - hmin) / h0 < 1.0e-9);
}

//------------------------------------------------------------------------------
// Elliptic orbit: after one period, Keplerian elements should be conserved.
BOOST_AUTO_TEST_CASE(EllipticOrbitElementsConserved)
{
  KeplerianElements el0{8000.0e3, 0.2, 45.0*DEG2RAD,
                        20.0*DEG2RAD, 30.0*DEG2RAD, 0.0};
  StateVector6 y0  = elements_to_state(el0);
  const double T   = orbital_period(el0.a);

  auto result = propagate(y0, 0.0, T,
                          MU_EARTH, 1.0e-9, 1.0e-10,
                          T / 200.0, false);

  const StateVector6& yf = result.states.back();
  KeplerianElements elf  = state_to_elements(yf);

  BOOST_TEST(elf.a == el0.a, boost::test_tools::tolerance(1e-3 / el0.a)); // 1 mm / a
  BOOST_TEST(elf.e == el0.e, boost::test_tools::tolerance(1e-8));
  BOOST_TEST(elf.i == el0.i, boost::test_tools::tolerance(1e-9));
}

//------------------------------------------------------------------------------
// n_steps recorded: we took at least one step
BOOST_AUTO_TEST_CASE(StepCountPositive)
{
  const double r = R_EARTH + 400.0e3;
  StateVector6 y0{r, 0.0, 0.0, 0.0, circular_speed(r), 0.0};
  const double T = orbital_period(r);

  auto result = propagate(y0, 0.0, T/10.0,
                          MU_EARTH, 1e-8, 1e-9, T/100.0, false);

  BOOST_TEST(result.n_steps > 0);
}

BOOST_AUTO_TEST_SUITE_END()
