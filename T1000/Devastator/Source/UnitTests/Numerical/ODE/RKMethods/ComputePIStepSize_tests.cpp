#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/CalculateNewYAndError.h"
#include "Numerical/ODE/RKMethods/CalculateScaledError.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "Numerical/ODE/RKMethods/ComputePIStepSize.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::CalculateNewYAndError;
using Numerical::ODE::RKMethods::CalculateScaledError;
using Numerical::ODE::RKMethods::ComputePIStepSize;
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
TEST(TestComputePIStepSize, ConstructsForDefaultValues)
{
  {
    ComputePIStepSize pi_step {alpha_5, beta_5};
  }
  {
    ComputePIStepSize pi_step {0.0875, 0.05};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestComputePIStepSize, ComputeNewStepSizeComputes)
{
  constexpr double epsilon {1.0e-6};

  ExampleSetupWithStdValarray<DOPRI5_s> setup {};

  CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>
    new_y_and_err {
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};

  CalculateScaledError scaled_error {epsilon, epsilon};

  ComputePIStepSize pi_step {alpha_5, beta_5};

  setup.y_out_ = new_y_and_err.calculate_new_y(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  auto calculated_error = new_y_and_err.calculate_error(
    setup.h_,
    setup.k_coefficients_);

  auto error = scaled_error.operator()<valarray<double>, 1>(
    setup.y_0_,
    setup.y_out_,
    calculated_error);

  EXPECT_DOUBLE_EQ(error, 10.047088008562323);

  double new_h {
    pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      setup.h_,
      false)};

  EXPECT_DOUBLE_EQ(new_h, 0.3257818497635847);

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<valarray<double>, 1>(
      setup.y_0_,
      setup.y_out_,
      calculated_error);

  EXPECT_DOUBLE_EQ(error, 1.517043286161648);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 0.2765856717210728);

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.t_0_ + new_h), 1e-6);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<valarray<double>, 1>(
      setup.y_0_,
      setup.y_out_,
      calculated_error);

  EXPECT_DOUBLE_EQ(error, 0.7192693159670872);

  setup.x_n_ += new_h;

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 0.12476919493370708);  

  setup.previous_error_ = error;

  // Step 2

  setup.y_n_ = setup.y_out_;

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_out_,
    setup.k_coefficients_.get_ith_coefficient(7),
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.x_n_ + new_h), 1e-6);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<valarray<double>, 1>(
      setup.y_n_,
      setup.y_out_,
      calculated_error);

  EXPECT_DOUBLE_EQ(error, 0.01123118086582314);

  setup.x_n_ += new_h;

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 0.20504144131432045);  

  setup.previous_error_ = error;

  // Step 3

  setup.y_n_ = setup.y_out_;

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_out_,
    setup.k_coefficients_.get_ith_coefficient(7),
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.x_n_ + new_h), 1e-6);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<valarray<double>, 1>(
      setup.y_n_,
      setup.y_out_,
      calculated_error);

  EXPECT_DOUBLE_EQ(error, 0.09858701685826891);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 0.1782310754175909);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestComputePIStepSize,
  ComputeNewStepSizeComputesWithScaledErrorOfZeroATolerance)
{
  ExampleSetupWithNVector<DOPRI5_s, 1> setup {};

  CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>
    new_y_and_err {
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};

  CalculateScaledError scaled_error {0.0, epsilon};

  ComputePIStepSize pi_step {alpha_5, beta_5};

  setup.x_n_ = setup.t_0_;
  setup.y_n_ = setup.y_0_;

  setup.y_out_ = new_y_and_err.calculate_new_y(
    setup.h_,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  auto calculated_error = new_y_and_err.calculate_error(
    setup.h_,
    setup.k_coefficients_);

  auto error = scaled_error.operator()<NVector<1>, 1>(
    setup.y_n_,
    setup.y_out_,
    calculated_error);

  EXPECT_DOUBLE_EQ(error, 17.094490672479946);

  double new_h {
    pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      setup.h_,
      false)};

  EXPECT_DOUBLE_EQ(new_h, 0.30242149125537182);

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<NVector<1>, 1>(
    setup.y_n_,
    setup.y_out_,
    calculated_error);

  EXPECT_DOUBLE_EQ(error, 2.1434187904795894);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 0.24462462635573054);

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<NVector<1>, 1>(
    setup.y_n_,
    setup.y_out_,
    calculated_error);

  EXPECT_DOUBLE_EQ(error, 0.8562538508592878);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.x_n_ + new_h), 1e-7);

  setup.x_n_ += new_h;

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 0.10769073212546879);  

  setup.previous_error_ = error;

  // Step 2

  setup.y_n_ = setup.y_out_;

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_out_,
    setup.k_coefficients_.get_ith_coefficient(7),
    setup.k_coefficients_);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<NVector<1>, 1>(
      setup.y_n_,
      setup.y_out_,
      calculated_error);

  EXPECT_DOUBLE_EQ(error, 0.010974841057994985);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.x_n_ + new_h), 1e-7);

  setup.x_n_ += new_h;

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 0.1800417760399134);  

  setup.previous_error_ = error;

  // Step 3

  setup.y_n_ = setup.y_out_;

  setup.y_out_ = new_y_and_err.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_out_,
    setup.k_coefficients_.get_ith_coefficient(7),
    setup.k_coefficients_);

  EXPECT_NEAR(setup.y_out_[0], exact_solution(setup.x_n_ + new_h), 1e-6);

  calculated_error = new_y_and_err.calculate_error(
      new_h,
      setup.k_coefficients_);

  error = scaled_error.operator()<NVector<1>, 1>(
      setup.y_n_,
      setup.y_out_,
      calculated_error);

  EXPECT_DOUBLE_EQ(error, 0.09620632689256294);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 0.15674696777373567);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests