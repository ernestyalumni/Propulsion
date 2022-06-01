#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/CalculateError.h"
#include "Numerical/ODE/RKMethods/CalculateNewY.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::CalculateError;
using Numerical::ODE::RKMethods::CalculateNewY;
using std::valarray;

template <size_t N>
using NVector = Algebra::Modules::Vectors::NVector<N>;

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
TEST(TestCalculateNewY, ConstructsWithStdValarray)
{
  const valarray a_tolerance (epsilon, 1);
  const valarray r_tolerance (2 * epsilon, 1);
  const valarray a_tolerance_2 (epsilon, 2);
  EXPECT_EQ(a_tolerance.size(), 1);
  EXPECT_EQ(a_tolerance_2.size(), 2);
  EXPECT_DOUBLE_EQ(a_tolerance[0], epsilon);
  EXPECT_DOUBLE_EQ(r_tolerance[0], 2 * epsilon);
  EXPECT_DOUBLE_EQ(a_tolerance_2[0], epsilon);
  EXPECT_DOUBLE_EQ(a_tolerance_2[1], epsilon);

  CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>
    calculate_error {
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      a_tolerance,
      r_tolerance};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewY, ConstructsWithNVector)
{
  const NVector<1> a_tolerance (epsilon);
  const NVector<1> r_tolerance (2 * epsilon);

  CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, NVector<1>>
    calculate_error {
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      a_tolerance,
      r_tolerance};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewY, CalculateErrorWithStdValarray)
{
  ExampleSetupWithStdValarray<DOPR853_s> setup {};

  CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray)> new_y {
    example_f_with_std_valarray,
    DOPR853_a_coefficients,
    DOPR853_b_coefficients,
    DOPR853_c_coefficients};

  CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>
    calculate_error {
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      valarray<double>(epsilon, 1),
      valarray<double>(2 * epsilon, 1)};

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  auto result = calculate_error.calculate_scaled_error<1>(
    setup.y_0_,
    setup.y_out_,
    setup.k_coefficients_,
    setup.h_);

  EXPECT_DOUBLE_EQ(result, 2554.8954356322129);

  auto y_in = setup.y_out_;

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + setup.h_, setup.y_out_),
    setup.k_coefficients_);

  result = calculate_error.calculate_scaled_error<1>(
    y_in,
    setup.y_out_,
    setup.k_coefficients_,
    setup.h_);

  EXPECT_DOUBLE_EQ(result, 2271.9490612109548);

  y_in = setup.y_out_;

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + 2 * setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + 2 * setup.h_, setup.y_out_),
    setup.k_coefficients_);

  result = calculate_error.calculate_scaled_error<1>(
    y_in,
    setup.y_out_,
    setup.k_coefficients_,
    setup.h_);

  EXPECT_DOUBLE_EQ(result, 1920.9197027842517);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewY, CalculateErrorWithNVector)
{
  ExampleSetupWithNVector<DOPR853_s, 1> setup {};

  CalculateNewY<DOPR853_s, decltype(example_f_with_NVector<1>)> new_y {
    example_f_with_NVector<1>,
    DOPR853_a_coefficients,
    DOPR853_b_coefficients,
    DOPR853_c_coefficients};

  CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, NVector<1>>
    calculate_error {
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      NVector<1>(epsilon),
      NVector<1>(2 * epsilon)};

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  auto result = calculate_error.calculate_scaled_error<1>(
    setup.y_0_,
    setup.y_out_,
    setup.k_coefficients_,
    setup.h_);

  EXPECT_DOUBLE_EQ(result, 2554.8954356322129);

  auto y_in = setup.y_out_;

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + setup.h_, setup.y_out_),
    setup.k_coefficients_);

  result = calculate_error.calculate_scaled_error<1>(
    y_in,
    setup.y_out_,
    setup.k_coefficients_,
    setup.h_);

  EXPECT_DOUBLE_EQ(result, 2271.9490612109548);

  y_in = setup.y_out_;

  setup.y_out_ = new_y.calculate_new_y(
    setup.h_,
    setup.t_0_ + 2 * setup.h_,
    setup.y_out_,
    new_y.calculate_derivative(setup.t_0_ + 2 * setup.h_, setup.y_out_),
    setup.k_coefficients_);

  result = calculate_error.calculate_scaled_error<1>(
    y_in,
    setup.y_out_,
    setup.k_coefficients_,
    setup.h_);

  EXPECT_DOUBLE_EQ(result, 1920.9197027842517);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests