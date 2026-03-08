#include "Numerical/ODE/RKMethods/Coefficients/ACoefficients.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"

#include "gtest/gtest.h"

using Numerical::ODE::RKMethods::Coefficients::ACoefficients;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace Coefficients
{

// cf. https://stackoverflow.com/questions/35533600/changing-variable-name-inside-a-loop
const auto& DOPRI5_a_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::a_coefficients;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestACoefficients, ConstructsFromInitializerList)
{
  // cf. Table 1.1. Low order Runge-Kutta methods, II.1 The First Runge-Kutta
  // Methods, pp. 135, Ordinary Differential Equations, Vol. 1, Nonstiff
  // Problems.

  // Runge, order 2
  {
    ACoefficients<2> as {0.5};

    EXPECT_EQ(as.size(), 1);
    EXPECT_EQ(as[0], 0.5);
  }
  // Runge, order 3, s = 4
  {
    ACoefficients<4> as {0.5, 0.0, 1.0, 0.0, 0.0, 1.0};

    EXPECT_EQ(as.size(), 6);
    EXPECT_EQ(as[0], 0.5);
    EXPECT_EQ(as[1], 0.0);
    EXPECT_EQ(as[2], 1.0);
    EXPECT_EQ(as[3], 0.0);
    EXPECT_EQ(as[5], 1.0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestDOPRI5Coefficients, GetsValueFromInheritedAccessOperator)
{
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients[0], 0.2);
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients[1], 3.0 / 40.0);
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients[2], 9.0 / 40.0);
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients[20], 11.0 / 84.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestDOPRI5Coefficients, GetsValueFromInheritedAtOperator)
{
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients.at(0), 0.2);
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients.at(1), 3.0 / 40.0);
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients.at(2), 9.0 / 40.0);
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients.at(20), 11.0 / 84.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestDOPRI5Coefficients, BeginAndEndIteratorsIterateThroughValues)
{
  std::size_t i {0};
  for (
    auto iter {DOPRI5_a_coefficients.begin()};
    iter != DOPRI5_a_coefficients.end();
    ++iter)
  {
    EXPECT_DOUBLE_EQ(*iter, DOPRI5_a_coefficients.at(i));
    ++i;
  }
}

} // namespace Coefficients
} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests