#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/CalculateNewYAndError.h"
#include "Numerical/ODE/RKMethods/CalculateScaledError.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "Numerical/ODE/RKMethods/ComputePIStepSize.h"
#include "Numerical/ODE/RKMethods/StepWithPIControl.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::CalculateNewYAndError;
using Numerical::ODE::RKMethods::CalculateScaledError;
using Numerical::ODE::RKMethods::ComputePIStepSize;
using Numerical::ODE::RKMethods::StepWithPIControl;
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestStepWithPIControl, ConstructsWithLValueObjects)
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

    StepWithPIControl step {new_y_and_err, scaled_error, pi_step};
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

    StepWithPIControl step {new_y_and_err, scaled_error, pi_step};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestStepWithPIControl, ConstructsWithRValueObjects)
{
  {
    StepWithPIControl step {
      CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
        example_f_with_std_valarray,
        DOPRI5_a_coefficients,
        DOPRI5_c_coefficients,
        DOPRI5_delta_coefficients},
      CalculateScaledError{epsilon, 2 * epsilon},
      ComputePIStepSize{alpha_5, beta_5}};
  }
  {
    StepWithPIControl step {
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
TEST(TestStepWithPIControl, StepWorksOnStdValarrayInputs)
{
  StepWithPIControl step {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{epsilon, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  auto inputs = step_inputs_with_std_valarray;

  auto h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.2765856717210728);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.12476919493370708);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.2765856717210728);

  h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.12476919493370708);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.20504144131432045);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.2765856717210728 + h_n);

  h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.20504144131432045);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.1782310754175909);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.40135486665477987 + h_n);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestStepWithPIControl, StepWorksOnNVectorInputs)
{
  StepWithPIControl step {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{epsilon, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  auto inputs = step_inputs_with_nvector;

  auto h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.2765856717210728);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.12476919493370708);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.2765856717210728);

  h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.12476919493370708);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.20504144131432045);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.2765856717210728 + h_n);

  h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.20504144131432045);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.1782310754175909);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.40135486665477987 + h_n);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestStepWithPIControl, StepWorksOnStdValarrayInputsWithZeroATolerance)
{
  StepWithPIControl step {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  auto inputs = step_inputs_with_std_valarray;

  auto h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.24462462635573054);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.10769073212546879);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.24462462635573054);

  h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.10769073212546879);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.1800417760399134);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.24462462635573054 + h_n);

  h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.1800417760399134);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.15674696777373567);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.53235713452111266);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestStepWithPIControl, StepWorksOnNVectorInputsWithZeroATolerance)
{
  StepWithPIControl step {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  auto inputs = step_inputs_with_nvector;

  auto h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.24462462635573054);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.10769073212546879);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.24462462635573054);

  h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.10769073212546879);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.1800417760399134);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.24462462635573054 + h_n);

  h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.1800417760399134);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.15674696777373567);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.53235713452111266);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestStepWithPIControl, StepWorksOnStdValarrayInputsWithLargerATolerance)
{
  StepWithPIControl step {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, larger_epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  auto inputs = step_inputs_with_std_valarray;

  auto h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.5);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-5);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.525548318135207);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.5);

  h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.52554831813520697);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-4);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.7673690892236448);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.5 + h_n);

  h_n = step.step<1, valarray<double>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.7673690892236448);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-4);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 1.0516696430816863);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 1.7929174073588516);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestStepWithPIControl, StepWorksOnNVectorInputsWithLargerATolerance)
{
  StepWithPIControl step {
    CalculateNewYAndError<DOPRI5_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients},
    CalculateScaledError{0.0, larger_epsilon},
    ComputePIStepSize{alpha_5, beta_5}};

  auto inputs = step_inputs_with_nvector;

  auto h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.5);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-5);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.525548318135207);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.5);

  h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.52554831813520697);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-4);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.7673690892236448);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.5 + h_n);

  h_n = step.step<1, NVector<1>>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.7673690892236448);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-4);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 1.0516696430816863);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 1.7929174073588516);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests