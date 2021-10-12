#include "Numerical/ODE/RhsVanDerPol.h"

#include "VanDerPolTestSetup.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <array>

using GoogleUnitTests::Numerical::ODE::VanDerPol::y1_in_20_steps;
using GoogleUnitTests::Numerical::ODE::VanDerPol::y2_in_20_steps;
using GoogleUnitTests::Numerical::ODE::VanDerPol::y1_result_20_steps;
using GoogleUnitTests::Numerical::ODE::VanDerPol::y2_result_20_steps;
using Numerical::ODE::RhsVanDerPol;
using std::array;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RhsVanDerPolTests, ConstructsFromEpsConstant)
{
  RhsVanDerPol<array<double, 2>> rhs_van {1.0e-3};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RhsVanDerPolTests, UpdatesDerivatives)
{
  RhsVanDerPol<array<double, 2>> rhs_van {1.0};
  array<double, 2> dydx {0., 0.};
  array<double, 2> y {2., 0.};

  constexpr double acceptable_error {1.0e-7};

  EXPECT_EQ(y1_in_20_steps.size(), 20);
  EXPECT_EQ(y2_in_20_steps.size(), 20);
  EXPECT_EQ(y1_result_20_steps.size(), 20);
  EXPECT_EQ(y2_result_20_steps.size(), 20);

  for (std::size_t i {0}; i < y1_in_20_steps.size(); ++i)
  {
    y[0] = y1_in_20_steps[i];
    y[1] = y2_in_20_steps[i];

    rhs_van(42.0, y, dydx);

    EXPECT_NEAR(y[1], dydx[0], acceptable_error);
    EXPECT_NEAR(dydx[0], y1_result_20_steps[i], acceptable_error);
    EXPECT_NEAR(dydx[1], y2_result_20_steps[i], acceptable_error);
  }
}

} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests