#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/CalculateError.h"
#include "Numerical/ODE/RKMethods/CalculateNewY.h"
#include "Numerical/ODE/RKMethods/CalculateNewYAndError.h"
#include "Numerical/ODE/RKMethods/CalculateScaledError.h"
#include "Numerical/ODE/RKMethods/ComputePIStepSize.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::CalculateError;
using Numerical::ODE::RKMethods::CalculateNewY;
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

constexpr double beta_8 {0.4 / 8};

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
TEST(TestComputePIStepSize, ComputeNewStepSizeComputesWithDOPR853)
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

  ComputePIStepSize pi_step {1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0};

  setup.x_n_ = setup.t_0_;
  setup.y_n_ = setup.y_0_;
  double new_h {0.000125};

  // Step 1.

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  auto error = calculate_error.calculate_scaled_error<1>(
    setup.y_0_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 1.229773621240583);

  new_h = pi_step.compute_new_step_size(
    error,
    setup.previous_error_,
    new_h,
    false);

  EXPECT_DOUBLE_EQ(new_h, 0.00011076833671953456);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 1.0897830910504656);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 9.9050720510786676e-05);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 0.97451763517510881);

  setup.x_n_ += new_h;

  EXPECT_DOUBLE_EQ(setup.x_n_, new_h);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 5.6356098659731305e-05);

  // Step 2.

  setup.y_n_ = setup.y_out_;
  auto dydx_n = setup.k_coefficients_.get_ith_coefficient(12);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    dydx_n,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 0.55447157681078107);

  setup.x_n_ += new_h;

  EXPECT_DOUBLE_EQ(setup.x_n_, 0.00015540681917051798);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 3.3449719755678483e-05);

  // Step 3.

  setup.y_n_ = setup.y_out_;
  dydx_n = setup.k_coefficients_.get_ith_coefficient(12);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    dydx_n,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 0.329104282396947);

  setup.x_n_ += new_h;

  EXPECT_DOUBLE_EQ(setup.x_n_, 0.00018885653892619647);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 2.0645955075026511e-05);
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestComputePIStepSize,
  ComputeNewStepSizeComputesWithScaledErrorOfZeroAToleranceAndDOPR853)
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
      NVector<1>(0.0),
      NVector<1>(epsilon)};

  ComputePIStepSize pi_step {1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0};

  setup.x_n_ = setup.t_0_;
  setup.y_n_ = setup.y_0_;
  double new_h {0.00003125};

  // Step 1.

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  auto error = calculate_error.calculate_scaled_error<1>(
    setup.y_0_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 1.2298889143184415);

  new_h = pi_step.compute_new_step_size(
    error,
    setup.previous_error_,
    new_h,
    false);

  EXPECT_DOUBLE_EQ(new_h, 2.7691889476686576e-05);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 1.0898659651280267);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 2.4762364794476034e-05);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    setup.dydx_0_,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 0.97457762136244164);

  setup.x_n_ += new_h;

  EXPECT_DOUBLE_EQ(setup.x_n_, new_h);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      true);

  EXPECT_DOUBLE_EQ(new_h, 1.4088780211807704e-05);

  // Step 2.

  setup.y_n_ = setup.y_out_;
  auto dydx_n = setup.k_coefficients_.get_ith_coefficient(12);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    dydx_n,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 0.55448539401722419);

  setup.x_n_ += new_h;

  EXPECT_DOUBLE_EQ(setup.x_n_, 3.8851145006283739e-05);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 8.3622692170408084e-06);

  // Step 3.

  setup.y_n_ = setup.y_out_;
  dydx_n = setup.k_coefficients_.get_ith_coefficient(12);

  setup.y_out_ = new_y.calculate_new_y(
    new_h,
    setup.x_n_,
    setup.y_n_,
    dydx_n,
    setup.k_coefficients_);

  error = calculate_error.calculate_scaled_error<1>(
    setup.y_n_,
    setup.y_out_,
    setup.k_coefficients_,
    new_h);

  EXPECT_DOUBLE_EQ(error, 0.32910621674343121);

  setup.x_n_ += new_h;

  EXPECT_DOUBLE_EQ(setup.x_n_, 4.7213414223324549e-05);

  new_h = pi_step.compute_new_step_size(
      error,
      setup.previous_error_,
      new_h,
      false);

  EXPECT_DOUBLE_EQ(new_h, 5.1613872921863078e-06);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests