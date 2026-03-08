#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/CalculateError.h"
#include "Numerical/ODE/RKMethods/CalculateNewY.h"
#include "Numerical/ODE/RKMethods/ComputePIStepSize.h"
#include "Numerical/ODE/RKMethods/HigherOrderStepWithPIControl.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::CalculateError;
using Numerical::ODE::RKMethods::CalculateNewY;
using Numerical::ODE::RKMethods::ComputePIStepSize;
using Numerical::ODE::RKMethods::HigherOrderStepWithPIControl;
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

constexpr double beta_8 {0.4 / 8};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderStepWithPIControl, ConstructsWithLValueObjects)
{
  {
    CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray)> new_y {
        example_f_with_std_valarray,
        DOPR853_a_coefficients,
        DOPR853_b_coefficients,
        DOPR853_c_coefficients};

    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>
      error {
        DOPR853_delta_coefficients,
        DOPR853_bhh_coefficients,
        valarray<double>(epsilon, 1),
        valarray<double>(2 * epsilon, 1)};

    ComputePIStepSize pi_step {1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0};

    HigherOrderStepWithPIControl step {new_y, error, pi_step};
  }
  {
    CalculateNewY<DOPR853_s, decltype(example_f_with_NVector<1>)> new_y {
      example_f_with_NVector<1>,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients};

    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, NVector<1>>
      error {
        DOPR853_delta_coefficients,
        DOPR853_bhh_coefficients,
        NVector<1>(epsilon),
        NVector<1>(2 * epsilon)};

    ComputePIStepSize pi_step {alpha_5, beta_5};

    HigherOrderStepWithPIControl step {new_y, error, pi_step};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderStepWithPIControl, ConstructsWithRValueObjects)
{
  {
    HigherOrderStepWithPIControl step {
      CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray)>{
        example_f_with_std_valarray,
        DOPR853_a_coefficients,
        DOPR853_b_coefficients,
        DOPR853_c_coefficients},
      CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>{
        DOPR853_delta_coefficients,
        DOPR853_bhh_coefficients,
        valarray<double>(epsilon, 1),
        valarray<double>(2 * epsilon, 1)},
      ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};
  }
  {
    HigherOrderStepWithPIControl step {
      CalculateNewY<DOPR853_s, decltype(example_f_with_NVector<1>)>{
        example_f_with_NVector<1>,
        DOPR853_a_coefficients,
        DOPR853_b_coefficients,
        DOPR853_c_coefficients},
      CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, NVector<1>>{
        DOPR853_delta_coefficients,
        DOPR853_bhh_coefficients,
        NVector<1>(epsilon),
        NVector<1>(2 * epsilon)},
      ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderStepWithPIControl, StepWorksOnStdValarrayInputs)
{
  HigherOrderStepWithPIControl step {
    CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients},
    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>{
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      valarray<double>(epsilon, 1),
      valarray<double>(epsilon, 1)},
    ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};

  auto inputs = step_inputs_with_std_valarray_for_DOPR853;

  auto h_n = step.step<1>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 7.2487358668285622e-05);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-15);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 4.1318288672540753e-05);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 7.2487358668285622e-05);

  h_n = step.step<1>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 4.1318288672540753e-05);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-15);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 3.8836372957621287e-05);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.00011380564734082637);

  h_n = step.step<1>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 3.8836372957621287e-05);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-15);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 3.5657211245225327e-05);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.00015264202029844766);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderStepWithPIControl, StepWorksOnLargerEpsilonAndNVector)
{
  HigherOrderStepWithPIControl step {
    CalculateNewY<DOPR853_s, decltype(example_f_with_NVector<1>)>{
      example_f_with_NVector<1>,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients},
    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, NVector<1>>{
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      NVector<1>(larger_epsilon),
      NVector<1>(larger_epsilon)},
    ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};

  auto inputs = step_inputs_with_nvector_for_DOPR853;

  auto h_n = step.step<1>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.5);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-10);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.30380934268061444);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.5);

  h_n = step.step<1>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.30380934268061444);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.28782462866297986);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 0.80380934268061444);

  h_n = step.step<1>(inputs);

  EXPECT_DOUBLE_EQ(h_n, 0.28782462866297986);
  EXPECT_NEAR(inputs.y_n_[0], exact_solution(inputs.x_n_), 1e-6);
  EXPECT_DOUBLE_EQ(inputs.h_n_, 0.27018280516715659);
  EXPECT_DOUBLE_EQ(inputs.x_n_, 1.0916339713435943);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests