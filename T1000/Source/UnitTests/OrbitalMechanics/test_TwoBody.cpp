//------------------------------------------------------------------------------
/// \file test_TwoBody.cpp
/// \brief Tests for two-body EOM and basic orbital mechanics helpers.
//------------------------------------------------------------------------------
#include "OrbitalMechanics/TwoBody.hpp"
#include "OrbitalMechanics/Constants.hpp"
#include "OrbitalMechanics/StateVector.hpp"

#include <boost/test/unit_test.hpp>
#include <cmath>

using namespace OrbitalMechanics;
using namespace OrbitalMechanics::Constants;

BOOST_AUTO_TEST_SUITE(TwoBodyTests)

//------------------------------------------------------------------------------
// EOM at a known point: circular orbit in x-y plane at altitude 400 km
// r = R_EARTH + 400e3  →  acceleration should be purely centripetal (-x direction
// for a satellite at [r,0,0] moving in +y)
BOOST_AUTO_TEST_CASE(AccelerationDirectionCircularOrbit)
{
  const double r = R_EARTH + 400.0e3;      // ~6.778e6 m
  const double vc = circular_speed(r);     // sqrt(μ/r)

  // Satellite at x=r, moving in +y direction
  StateVector6 y{r, 0.0, 0.0, 0.0, vc, 0.0};

  TwoBodyEOM eom;
  StateVector6 dydt = eom(0.0, y);

  // Position derivative should equal velocity
  BOOST_TEST(dydt[0] == 0.0,  boost::test_tools::tolerance(1e-12));
  BOOST_TEST(dydt[1] == vc,   boost::test_tools::tolerance(1e-12));
  BOOST_TEST(dydt[2] == 0.0,  boost::test_tools::tolerance(1e-12));

  // Acceleration: ax = -μ/r², ay = 0, az = 0
  const double a_expected = -MU_EARTH / (r * r);
  BOOST_TEST(dydt[3] == a_expected, boost::test_tools::tolerance(1e-12));
  BOOST_TEST(dydt[4] == 0.0,        boost::test_tools::tolerance(1e-12));
  BOOST_TEST(dydt[5] == 0.0,        boost::test_tools::tolerance(1e-12));
}

//------------------------------------------------------------------------------
// Circular speed: v² = μ/r  →  centripetal = v²/r = μ/r²  (matches gravity)
BOOST_AUTO_TEST_CASE(CircularSpeedConsistency)
{
  const double r  = R_EARTH + 600.0e3;
  const double vc = circular_speed(r);
  BOOST_TEST(vc * vc / r == MU_EARTH / (r * r),
             boost::test_tools::tolerance(1e-10));
}

//------------------------------------------------------------------------------
// Orbital period of ISS-like orbit (400 km altitude) ≈ 92.7 minutes
BOOST_AUTO_TEST_CASE(OrbitalPeriodISSApprox)
{
  const double r = R_EARTH + 400.0e3;
  const double T = orbital_period(r);
  const double T_minutes = T / 60.0;

  // ISS period is ≈ 92.7 min
  BOOST_TEST(T_minutes > 91.0);
  BOOST_TEST(T_minutes < 94.0);
}

//------------------------------------------------------------------------------
// vis-viva at apoapsis of GTO (perigee 200 km, apogee 35786 km)
BOOST_AUTO_TEST_CASE(VisVivaGTOApoapsis)
{
  const double r_p = R_EARTH + 200.0e3;
  const double r_a = R_EARTH + 35786.0e3;
  const double a   = 0.5 * (r_p + r_a);

  const double v_apo = vis_viva(MU_EARTH, r_a, a);

  // Known GTO apoapsis speed ≈ 1.6 km/s
  BOOST_TEST(v_apo > 1500.0);
  BOOST_TEST(v_apo < 1700.0);
}

//------------------------------------------------------------------------------
// EOM is autonomous: same r → same acceleration regardless of t
BOOST_AUTO_TEST_CASE(EOMAusinomous)
{
  const double r = R_EARTH + 500.0e3;
  StateVector6 y{r, 0.0, 0.0, 0.0, circular_speed(r), 0.0};

  TwoBodyEOM eom;
  auto d1 = eom(0.0, y);
  auto d2 = eom(1000.0, y);

  for (int i = 0; i < 6; ++i)
    BOOST_TEST(d1[i] == d2[i]);
}

BOOST_AUTO_TEST_SUITE_END()
