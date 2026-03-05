//------------------------------------------------------------------------------
/// \file test_OrbitalElements.cpp
/// \brief Round-trip and physics tests for element ↔ state conversions.
//------------------------------------------------------------------------------
#include "OrbitalMechanics/OrbitalElements.hpp"
#include "OrbitalMechanics/Constants.hpp"
#include "OrbitalMechanics/StateVector.hpp"

#include <boost/test/unit_test.hpp>
#include <cmath>

using namespace OrbitalMechanics;
using namespace OrbitalMechanics::Constants;

BOOST_AUTO_TEST_SUITE(OrbitalElementsTests)

//------------------------------------------------------------------------------
// Kepler's equation: E - e*sin(E) = M.  Verify round-trip.
BOOST_AUTO_TEST_CASE(KeplerEquationRoundTrip)
{
  const double e = 0.5;
  for (double M_deg : {0.0, 30.0, 90.0, 150.0, 180.0, 270.0, 359.0})
  {
    const double M = M_deg * DEG2RAD;
    const double E = kepler_E_from_M(M, e);
    const double M_check = E - e * std::sin(E);
    BOOST_TEST(M_check == M, boost::test_tools::tolerance(1e-11));
  }
}

//------------------------------------------------------------------------------
// Circular ISS-like orbit (e≈0): state → elements → state round-trip
BOOST_AUTO_TEST_CASE(RoundTripCircularOrbit)
{
  const double r  = R_EARTH + 400.0e3;
  const double vc = std::sqrt(MU_EARTH / r);

  // Circular, equatorial-ish, slight inclination
  KeplerianElements el_in{r, 0.0, 51.6*DEG2RAD, 0.5, 0.0, 0.0};

  StateVector6 s    = elements_to_state(el_in);
  KeplerianElements el_out = state_to_elements(s);

  BOOST_TEST(el_out.a == el_in.a,    boost::test_tools::tolerance(1e-3));
  // e is zero-ish; just check it's small
  BOOST_TEST(el_out.e < 1e-9);
  BOOST_TEST(el_out.i == el_in.i,    boost::test_tools::tolerance(1e-9));
}

//------------------------------------------------------------------------------
// Elliptic orbit round-trip: a, e, i, Ω, ω, ν all preserved
BOOST_AUTO_TEST_CASE(RoundTripEllipticOrbit)
{
  KeplerianElements el_in{
    8000.0e3,    // a = 8000 km
    0.1,         // e
    45.0*DEG2RAD,// i
    30.0*DEG2RAD,// Ω
    60.0*DEG2RAD,// ω
    120.0*DEG2RAD// ν
  };

  StateVector6 s = elements_to_state(el_in);
  KeplerianElements el_out = state_to_elements(s);

  BOOST_TEST(el_out.a    == el_in.a,    boost::test_tools::tolerance(1e-3)); // 1 mm
  BOOST_TEST(el_out.e    == el_in.e,    boost::test_tools::tolerance(1e-10));
  BOOST_TEST(el_out.i    == el_in.i,    boost::test_tools::tolerance(1e-10));
  BOOST_TEST(el_out.raan == el_in.raan, boost::test_tools::tolerance(1e-10));
  BOOST_TEST(el_out.aop  == el_in.aop,  boost::test_tools::tolerance(1e-10));
  BOOST_TEST(el_out.nu   == el_in.nu,   boost::test_tools::tolerance(1e-10));
}

//------------------------------------------------------------------------------
// Energy from elements_to_state matches vis-viva
BOOST_AUTO_TEST_CASE(EnergyFromStateMatchesVisViva)
{
  KeplerianElements el{7000.0e3, 0.05, 30.0*DEG2RAD, 0.0, 0.0, 45.0*DEG2RAD};
  StateVector6 s = elements_to_state(el);

  const double eps_state = specific_energy(s, MU_EARTH);
  const double eps_theory = -MU_EARTH / (2.0 * el.a);

  BOOST_TEST(eps_state == eps_theory, boost::test_tools::tolerance(1e-6));
}

//------------------------------------------------------------------------------
// Angular momentum from circular orbit: |h| = sqrt(μ·p) = sqrt(μ·a) for e=0
BOOST_AUTO_TEST_CASE(AngularMomentumCircularOrbit)
{
  const double a = 7000.0e3;
  KeplerianElements el{a, 0.0, 0.3, 0.0, 0.0, 0.0};
  StateVector6 s = elements_to_state(el);

  const double h_state  = angular_momentum_mag(s);
  const double h_theory = std::sqrt(MU_EARTH * a); // p = a for e=0

  BOOST_TEST(h_state == h_theory, boost::test_tools::tolerance(1e-6));
}

//------------------------------------------------------------------------------
// GTO: periapsis 200 km, apoapsis 35786 km (geostationary transfer orbit)
BOOST_AUTO_TEST_CASE(GTOEccentricityAndSemiMajorAxis)
{
  const double r_p = R_EARTH + 200.0e3;
  const double r_a = R_EARTH + 35786.0e3;
  const double a   = 0.5 * (r_p + r_a);
  const double e   = (r_a - r_p) / (r_a + r_p);

  KeplerianElements el{a, e, 27.0*DEG2RAD, 0.0, 0.0, 0.0};  // at periapsis
  StateVector6 s = elements_to_state(el);
  KeplerianElements el_out = state_to_elements(s);

  BOOST_TEST(el_out.a == a, boost::test_tools::tolerance(1e-1)); // 10 cm
  BOOST_TEST(el_out.e == e, boost::test_tools::tolerance(1e-9));
}

BOOST_AUTO_TEST_SUITE_END()
