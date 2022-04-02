#include "Numerical/ODE/RKMethods/Coefficients/ACoefficients.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"

#include "gtest/gtest.h"

using Numerical::ODE::RKMethods::Coefficients::ACoefficients;

const auto DOPRI5_a_coefficients =
  Numerical::ODE::RKMethods::DOPRI5Coefficients::a_coefficients;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestDOPRI5Coefficients, GetsValueFromInheritedAccessOperator)
{
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients[0], 0.2);
  EXPECT_DOUBLE_EQ(DOPRI5_a_coefficients[1], 3.0 / 40.0);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests