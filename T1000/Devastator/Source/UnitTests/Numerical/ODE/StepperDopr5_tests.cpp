#include "Numerical/ODE/StepperDopr5.h"

#include "gtest/gtest.h"

#include <vector>

using Numerical::ODE::StdFunctionDerivativeType;
using Numerical::ODE::StepperDopr5;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

TEST(StepperDopr5Tests, Constructs)
{
  vector<double> y_0 {0.5};
  vector<double> dydx_0 {1.5};
  double x_0 {0};
  const double a_tolerance {0.0001};
  const double r_tolerance {0.001};

  StepperDopr5<StdFunctionDerivativeType> sd {
    y_0,
    dydx_0,
    x_0,
    a_tolerance,
    r_tolerance,
    true};

  EXPECT_EQ(sd.x_, x_0);
  EXPECT_EQ(sd.y_, y_0);
  EXPECT_EQ(sd.dydx_, dydx_0);
  EXPECT_EQ(sd.a_tolerance_, a_tolerance);
  EXPECT_EQ(sd.r_tolerance_, r_tolerance);
  EXPECT_EQ(sd.dense_, true);
  EXPECT_EQ(sd.n_, y_0.size());
  EXPECT_EQ(sd.n_eqns_, y_0.size());
}

} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests