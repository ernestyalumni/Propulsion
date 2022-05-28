#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/CalculateNewYAndError.h"
#include "Numerical/ODE/RKMethods/CalculateScaledError.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "Numerical/ODE/RKMethods/ComputePIStepSize.h"
#include "Numerical/ODE/RKMethods/IntegrateWithPIControl.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::CalculateNewYAndError;
using Numerical::ODE::RKMethods::CalculateScaledError;
using Numerical::ODE::RKMethods::ComputePIStepSize;
using Numerical::ODE::RKMethods::IntegrateWithPIControl;
using Numerical::ODE::RKMethods::IntegrationInputsForDenseOutput;
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

constexpr double larger_epsilon {1e-2};

const IntegrationInputsForDenseOutput dense_inputs_for_std_valarray {
  step_inputs_with_std_valarray.y_n_,
  0.0,
  2.0,
  5};

const IntegrationInputsForDenseOutput dense_inputs_for_nvector {
  step_inputs_with_nvector.y_n_,
  0.0,
  2.0,
  5};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl, ConstructsWithLValueObjects)
{
  {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>
      new_y_and_err {
        example_f_with_std_valarray,
        DOPRI5_a_coefficients,
        DOPRI5_c_coefficients,
        DOPRI5_delta_coefficients};

    CalculateScaledError scaled_error {epsilon, 2 * epsilon};

    ComputePIStepSize pi_step {alpha_5, beta_5};

    IntegrateWithPIControl integrate {new_y_and_err, scaled_error, pi_step};
  }
  {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>
      new_y_and_err {
        example_f_with_NVector<1>,
        DOPRI5_a_coefficients,
        DOPRI5_c_coefficients,
        DOPRI5_delta_coefficients};

    CalculateScaledError scaled_error {epsilon, 2 * epsilon};

    ComputePIStepSize pi_step {alpha_5, beta_5};

    IntegrateWithPIControl integrate {new_y_and_err, scaled_error, pi_step};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl, ConstructsWithRValueObjects)
{
  {
    IntegrateWithPIControl integrate {
      CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
        example_f_with_std_valarray,
        DOPRI5_a_coefficients,
        DOPRI5_c_coefficients,
        DOPRI5_delta_coefficients},
      CalculateScaledError{epsilon, 2 * epsilon},
      ComputePIStepSize{alpha_5, beta_5}      
      };
  }
  {
    IntegrateWithPIControl integrate {
      CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
        example_f_with_NVector<1>,
        DOPRI5_a_coefficients,
        DOPRI5_c_coefficients,
        DOPRI5_delta_coefficients},
      CalculateScaledError{epsilon, 2 * epsilon},
      ComputePIStepSize{alpha_5, beta_5}};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl, IntegrateIntegratesWithStdValarray)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{epsilon, 2 * epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate<1, valarray<double>>(
    integrate_inputs_with_std_valarray);  

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 11);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.2);
  EXPECT_DOUBLE_EQ(result_x[2], 0.31741001868168683);
  EXPECT_DOUBLE_EQ(result_x[3], 0.49769643876973546);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.207196030542153);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-6);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_DOUBLE_EQ(result_h[i], result_x[i + 1] - result_x[i]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl, IntegrateIntegratesWithNVector)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{epsilon, 2 * epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate<1, NVector<1>>(
    integrate_inputs_with_nvector);  

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 11);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.2);
  EXPECT_DOUBLE_EQ(result_x[2], 0.31741001868168683);
  EXPECT_DOUBLE_EQ(result_x[3], 0.49769643876973546);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 1.9331867437502506);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.207196030542153);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-6);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_DOUBLE_EQ(result_h[i], result_x[i + 1] - result_x[i]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateIntegratesWithZeroAToleranceWithStdValarray)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate<1, valarray<double>>(
    integrate_inputs_with_std_valarray);  

  const auto result_x {std::get<0>(result)};
  const auto result_y {std::get<1>(result)};
  const auto result_h {std::get<2>(result)};

  EXPECT_EQ(result_x.size(), 12);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.2);
  EXPECT_DOUBLE_EQ(result_x[2], 0.29974091152182608);
  EXPECT_DOUBLE_EQ(result_x[3], 0.46072162525238469);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 1.9208946571455878);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.1671462301581483);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-6);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_DOUBLE_EQ(result_h[i], result_x[i + 1] - result_x[i]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateIntegratesWithZeroAToleranceWithNVector)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate<1, NVector<1>>(
    integrate_inputs_with_nvector);  

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 12);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.2);
  EXPECT_DOUBLE_EQ(result_x[2], 0.29974091152182608);
  EXPECT_DOUBLE_EQ(result_x[3], 0.46072162525238469);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 1.9208946571455878);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.1671462301581483);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-5);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_DOUBLE_EQ(result_h[i], result_x[i + 1] - result_x[i]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateIntegratesWithLargerAToleranceWithStdValarray)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, larger_epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate<1, valarray<double>>(
    integrate_inputs_with_std_valarray);  

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 6);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.2);
  EXPECT_DOUBLE_EQ(result_x[2], 0.56213736136341397);
  EXPECT_DOUBLE_EQ(result_x[3], 1.0470727466176584);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 1.704963733149909);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.6693284571058449);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-3);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_DOUBLE_EQ(result_h[i], result_x[i + 1] - result_x[i]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateIntegratesWithLargerAToleranceWithNVector)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, larger_epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate<1, NVector<1>>(
    integrate_inputs_with_nvector);  

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 6);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.2);
  EXPECT_DOUBLE_EQ(result_x[2], 0.56213736136341397);
  EXPECT_DOUBLE_EQ(result_x[3], 1.0470727466176584);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 1.704963733149909);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.6693284571058449);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-3);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_DOUBLE_EQ(result_h[i], result_x[i + 1] - result_x[i]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrationInputsForDenseOutput, Constructs)
{
  {
    EXPECT_EQ(dense_inputs_for_std_valarray.h_, 0.4);
    EXPECT_EQ(dense_inputs_for_std_valarray.x_.size(), 6);

    double x {dense_inputs_for_std_valarray.x_1_};
    for (
      auto iter = dense_inputs_for_std_valarray.x_.begin();
      iter != dense_inputs_for_std_valarray.x_.end();
      ++iter)
    {
      EXPECT_DOUBLE_EQ(*iter, x);
      x += dense_inputs_for_std_valarray.h_;
    }

    EXPECT_DOUBLE_EQ(dense_inputs_for_std_valarray.x_[4], 1.6);
    EXPECT_DOUBLE_EQ(dense_inputs_for_std_valarray.x_[5], 2.0);
  }
  {
    EXPECT_EQ(dense_inputs_for_nvector.h_, 0.4);
    EXPECT_EQ(dense_inputs_for_nvector.x_.size(), 6);

    double x {dense_inputs_for_nvector.x_1_};
    for (
      auto iter = dense_inputs_for_nvector.x_.begin();
      iter != dense_inputs_for_nvector.x_.end();
      ++iter)
    {
      EXPECT_DOUBLE_EQ(*iter, x);
      x += dense_inputs_for_nvector.h_;
    }

    EXPECT_DOUBLE_EQ(dense_inputs_for_nvector.x_[4], 1.6);
    EXPECT_DOUBLE_EQ(dense_inputs_for_nvector.x_[5], 2.0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl, IntegrateForDenseOutputWorksWithStdValarray)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{epsilon, 2 * epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate_for_dense_output<1, valarray<double>>(
    dense_inputs_for_std_valarray);

  const auto result_y = std::get<0>(result);
  const auto result_h = std::get<1>(result);

  EXPECT_EQ(result_y.size(), dense_inputs_for_std_valarray.x_.size());
  EXPECT_EQ(result_h.size(), dense_inputs_for_std_valarray.x_.size() - 1);

  for (std::size_t i {0}; i < dense_inputs_for_std_valarray.x_.size(); ++i)
  {
    EXPECT_NEAR(
      result_y[i][0],
      exact_solution(dense_inputs_for_std_valarray.x_[i]),
      1e-4);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl, IntegrateForDenseOutputWorksWithNVector)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{epsilon, 2 * epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate_for_dense_output<1, NVector<1>>(
    dense_inputs_for_nvector);

  const auto result_y = std::get<0>(result);
  const auto result_h = std::get<1>(result);

  EXPECT_EQ(result_y.size(), dense_inputs_for_nvector.x_.size());
  EXPECT_EQ(result_h.size(), dense_inputs_for_nvector.x_.size() - 1);

  for (std::size_t i {0}; i < dense_inputs_for_nvector.x_.size(); ++i)
  {
    EXPECT_NEAR(
      result_y[i][0],
      exact_solution(dense_inputs_for_nvector.x_[i]),
      1e-4);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateForDenseOutputWorksWithZeroAToleranceAndStdValarray)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate_for_dense_output<1, valarray<double>>(
    dense_inputs_for_std_valarray);

  const auto result_y = std::get<0>(result);
  const auto result_h = std::get<1>(result);

  EXPECT_EQ(result_y.size(), dense_inputs_for_std_valarray.x_.size());
  EXPECT_EQ(result_h.size(), dense_inputs_for_std_valarray.x_.size() - 1);

  for (std::size_t i {0}; i < dense_inputs_for_std_valarray.x_.size(); ++i)
  {
    EXPECT_NEAR(
      result_y[i][0],
      exact_solution(dense_inputs_for_std_valarray.x_[i]),
      1e-4);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateForDenseOutputWorksWithZeroAToleranceAndNVector)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate_for_dense_output<1, NVector<1>>(
    dense_inputs_for_nvector);

  const auto result_y = std::get<0>(result);
  const auto result_h = std::get<1>(result);

  EXPECT_EQ(result_y.size(), dense_inputs_for_nvector.x_.size());
  EXPECT_EQ(result_h.size(), dense_inputs_for_nvector.x_.size() - 1);

  for (std::size_t i {0}; i < dense_inputs_for_nvector.x_.size(); ++i)
  {
    EXPECT_NEAR(
      result_y[i][0],
      exact_solution(dense_inputs_for_nvector.x_[i]),
      1e-4);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateForDenseOutputWorksWithLargerAToleranceAndStdValarray)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, larger_epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate_for_dense_output<1, valarray<double>>(
    dense_inputs_for_std_valarray);

  const auto result_y = std::get<0>(result);
  const auto result_h = std::get<1>(result);

  EXPECT_EQ(result_y.size(), dense_inputs_for_std_valarray.x_.size());
  EXPECT_EQ(result_h.size(), dense_inputs_for_std_valarray.x_.size() - 1);

  for (std::size_t i {0}; i < dense_inputs_for_std_valarray.x_.size(); ++i)
  {
    EXPECT_NEAR(
      result_y[i][0],
      exact_solution(dense_inputs_for_std_valarray.x_[i]),
      1e-2);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestIntegrateWithPIControl,
  IntegrateForDenseOutputWorksWithLargerAToleranceAndNVector)
{
  IntegrateWithPIControl integrate {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, larger_epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  const auto result = integrate.integrate_for_dense_output<1, NVector<1>>(
    dense_inputs_for_nvector);

  const auto result_y = std::get<0>(result);
  const auto result_h = std::get<1>(result);

  EXPECT_EQ(result_y.size(), dense_inputs_for_nvector.x_.size());
  EXPECT_EQ(result_h.size(), dense_inputs_for_nvector.x_.size() - 1);

  for (std::size_t i {0}; i < dense_inputs_for_nvector.x_.size(); ++i)
  {
    EXPECT_NEAR(
      result_y[i][0],
      exact_solution(dense_inputs_for_nvector.x_[i]),
      1e-2);
  }
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests