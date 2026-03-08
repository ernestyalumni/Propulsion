#include "Numerical/ODE/RKMethods/CalculateNewY.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

using Numerical::ODE::RKMethods::CalculateNewY;

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
TEST(TestCalculateNewY, Constructs)
{
  {
    CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray)> new_y {
      example_f_with_std_valarray,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients};
  }
  {
    CalculateNewY<DOPR853_s, decltype(example_f_with_NVector<1>)> new_y {
      example_f_with_NVector<1>,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewY, CalculateNewYWithStdValarray)
{
  ExampleSetupWithStdValarray<DOPR853_s> setup {};

  CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray)> new_y {
    example_f_with_std_valarray,
    DOPR853_a_coefficients,
    DOPR853_b_coefficients,
    DOPR853_c_coefficients};

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.h_), 1e-10);

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + setup.h_, setup.y_out_),
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(2 * setup.h_), 1e-10);

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + 2 * setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + 2 * setup.h_, setup.y_out_),
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(3 * setup.h_), 1e-10);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewY, CalculateNewYWithNVector)
{
  ExampleSetupWithNVector<DOPR853_s, 1> setup {};

  CalculateNewY<DOPR853_s, decltype(example_f_with_NVector<1>)> new_y {
    example_f_with_NVector<1>,
    DOPR853_a_coefficients,
    DOPR853_b_coefficients,
    DOPR853_c_coefficients};

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.h_), 1e-10);

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + setup.h_, setup.y_out_),
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(2 * setup.h_), 1e-10);

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + 2 * setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + 2 * setup.h_, setup.y_out_),
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(3 * setup.h_), 1e-10);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests