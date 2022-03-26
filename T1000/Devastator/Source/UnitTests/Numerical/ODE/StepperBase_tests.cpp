#include "Numerical/ODE/StepperBase.h"

#include "gtest/gtest.h"

#include <vector>

using Numerical::ODE::StepperBase;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
/// \ref https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
//------------------------------------------------------------------------------
TEST(StepperBaseTests, Constructs)
{
  vector<double> y_0 {0.5};
  vector<double> dydx_0 {1.5};
  double x_0 {0};
  const double a_tolerance {0.0001};
  const double r_tolerance {0.001};

  StepperBase sb {y_0, dydx_0, x_0, a_tolerance, r_tolerance, true};

  EXPECT_EQ(sb.x_, x_0);
  EXPECT_EQ(sb.y_, y_0);
  EXPECT_EQ(sb.dydx_, dydx_0);
  EXPECT_EQ(sb.a_tolerance_, a_tolerance);
  EXPECT_EQ(sb.r_tolerance_, r_tolerance);
  EXPECT_EQ(sb.dense_, true);
  EXPECT_EQ(sb.n_, y_0.size());
  EXPECT_EQ(sb.n_eqns_, y_0.size());
}

} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests