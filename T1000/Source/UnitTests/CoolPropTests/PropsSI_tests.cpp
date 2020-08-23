//------------------------------------------------------------------------------
/// \file PropsSI_tests.cpp
//------------------------------------------------------------------------------
#include "CoolProp.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <utility>

using CoolProp::PropsSI;

BOOST_AUTO_TEST_SUITE(CoolPropTests)
BOOST_AUTO_TEST_SUITE(PropsSI_tests)

//------------------------------------------------------------------------------
/// \details Call to PropsSI function for pure fluids, pseudo-pure fluids, and
/// mixtures.
/// For humid air properties, see Humid air properties.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PropsSIOutputsProperty)
{
  //----------------------------------------------------------------------------
  /// Saturation temperature of Water at 1 atm in K
  /// \details First parameter, T, is *output* property that'll be returned by
  /// PropsSI.
  /// 2nd and 4th parameters are the specified input pair of properties that
  /// determine state point where output property calculated.
  /// 3rd and 5th parameters are values of input pair properties and determine
  /// state point.
  /// 6th and last parameter is the fluid for which output property will be
  /// calculated.
  ///
  /// For pure and pseudo-pure fluids, 2 state variables required to fix state.
  /// Equations of state based on T and rho as state variables, so T, rho always
  /// fastest inputs.
  /// P, T bit slower (3-10 times).
  ///
  /// cf. http://www.coolprop.org/coolprop/HighLevelAPI.html#parameter-table
  /// Table of string inputs to PropsSI function.
  /// Q mol/mol Mass vapor quality
  //----------------------------------------------------------------------------
  const auto water_temperature = PropsSI("T", "P", 101325, "Q", 0, "Water");

  BOOST_TEST((typeid(water_temperature) == typeid(double)));
  BOOST_TEST(water_temperature == 373.1242958476844);
}

BOOST_AUTO_TEST_SUITE(LiquidOxygen_tests)

const std::string fluid {"Oxygen"};

const std::pair<std::string, double> temperature1_K {
  std::make_pair<std::string, double>("T", 66.0)};

// Freezing is 54.36 so this is "near freezing."
const std::pair<std::string, double> freezingT_K {
  std::make_pair<std::string, double>("T", 54.37)};

const std::pair<std::string, double> boilingT_K {
  std::make_pair<std::string, double>("T", 90.19)};

const std::pair<std::string, double> atm_pressure_Pa {
  std::make_pair<std::string, double>("P", 101325)};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(PropsSIOutputsMassDensityProperties)
{
  double density_LOx = PropsSI(
    "D",
    atm_pressure_Pa.first,
    atm_pressure_Pa.second,
    temperature1_K.first,
    temperature1_K.second,
    fluid);

  // D, DMASS, Dmass kg/m^3 Mass density
  BOOST_TEST(density_LOx == 1255.3003769211227);

  density_LOx = PropsSI(
    "DMASS",
    atm_pressure_Pa.first,
    atm_pressure_Pa.second,
    temperature1_K.first,
    temperature1_K.second,
    fluid);

  BOOST_TEST(density_LOx == 1255.3003769211227);

  density_LOx = PropsSI(
    "Dmass",
    atm_pressure_Pa.first,
    atm_pressure_Pa.second,
    temperature1_K.first,
    temperature1_K.second,
    fluid);

  BOOST_TEST(density_LOx == 1255.3003769211227);

  density_LOx = PropsSI(
    "DMASS",
    atm_pressure_Pa.first,
    atm_pressure_Pa.second,
    freezingT_K.first,
    freezingT_K.second,
    fluid);

  BOOST_TEST(density_LOx == 1306.1523017346992);

  density_LOx = PropsSI(
    "DMASS",
    atm_pressure_Pa.first,
    atm_pressure_Pa.second,
    boilingT_K.first,
    boilingT_K.second,
    fluid);

  BOOST_TEST(density_LOx == 4.466991946130471);
}

BOOST_AUTO_TEST_SUITE_END() // LiquidOxygen_tests

BOOST_AUTO_TEST_SUITE_END() // PropsSI_tests
BOOST_AUTO_TEST_SUITE_END() // CoolPropTests
