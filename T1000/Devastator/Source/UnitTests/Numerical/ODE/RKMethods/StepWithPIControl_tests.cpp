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

  step.step<1, valarray<double>>(inputs);

  // TODO: check steps for x.
  // EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-7);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.12476919493370708);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.12476919493370708);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests