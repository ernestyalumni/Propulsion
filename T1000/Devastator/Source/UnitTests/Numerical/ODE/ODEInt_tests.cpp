#include "Numerical/ODE/ODEInt.h"

#include "ODEIntTestSetup.h"
#include "gtest/gtest.h"

#include <initializer_list>
#include <vector>

using Numerical::ODE::OdeInt;
using GoogleUnitTests::Numerical::ODE::ODEInt::TestStepper;
using std::vector;

class TestOdeInt : public ::testing::Test
{
  protected:

    static constexpr std::initializer_list<double> y_i_values {2.0, 0.0};

    TestOdeInt():
      y_i_{y_i_values},
      dydx_i_{},
      x_i_{},
      a_tolerance_{},
      r_tolerance_{},
      is_dense_{false},
      s_{y_i_, dydx_i_, x_i_, a_tolerance_, r_tolerance_, is_dense_}
    {}

    void SetUp() override
    {}

    vector<double> y_i_;
    vector<double> dydx_i_;
    double x_i_;
    double a_tolerance_;
    double r_tolerance_;
    bool is_dense_;

    TestStepper s_;
};

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST_F(TestOdeInt, Constructs)
{

  SUCCEED();
}


} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests