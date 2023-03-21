#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/CalculateError.h"
#include "Numerical/ODE/RKMethods/CalculateNewY.h"
#include "Numerical/ODE/RKMethods/ComputePIStepSize.h"
#include "Numerical/ODE/RKMethods/HigherOrderIntegrateWithPIControl.h"
#include "Numerical/ODE/RKMethods/IntegrationInputs.h"
#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::CalculateError;
using Numerical::ODE::RKMethods::CalculateNewY;
using Numerical::ODE::RKMethods::ComputePIStepSize;
using Numerical::ODE::RKMethods::HigherOrderIntegrateWithPIControl;
using Numerical::ODE::RKMethods::IntegrationInputs;
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

constexpr double larger_epsilon {1e-4};

constexpr double beta_8 {0.4 / 8};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderIntegrateWithPIControl, ConstructsWithLValueObjects)
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

    HigherOrderIntegrateWithPIControl integrate {new_y, error, pi_step};
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

    HigherOrderIntegrateWithPIControl integrate {new_y, error, pi_step};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderIntegrateWithPIControl, ConstructsWithRValueObjects)
{
  {
    HigherOrderIntegrateWithPIControl integrate {
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
    HigherOrderIntegrateWithPIControl integrate {
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
TEST(TestHigherOrderIntegrateWithPIControl, IntegrateIntegratesWithStdValarray)
{
  HigherOrderIntegrateWithPIControl integrate {
    CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray)>{
      example_f_with_std_valarray,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients},
    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>{
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      valarray<double>(larger_epsilon, 1),
      valarray<double>(larger_epsilon, 1)},
    ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};

  const auto result = integrate.integrate<1>(
    integrate_inputs_with_std_valarray);

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 12147);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.0073435040953447293);
  EXPECT_DOUBLE_EQ(result_x[2], 0.0115275567498708);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 1.9999875006224939);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.0002916820184558);
  EXPECT_DOUBLE_EQ(result_h[0], 0.0073435040953447293);
  EXPECT_DOUBLE_EQ(result_h[1], 0.0041840526545260711);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-13);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_NEAR(result_h[i], result_x[i + 1] - result_x[i], 1e-15);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderIntegrateWithPIControl, IntegrateIntegratesWithNVector)
{
  HigherOrderIntegrateWithPIControl integrate {
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

  const auto result = integrate.integrate<1>(integrate_inputs_with_nvector);

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 12147);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.0073435040953447293);
  EXPECT_DOUBLE_EQ(result_x[2], 0.0115275567498708);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 1.9999875006224939);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 2.0002916820184558);
  EXPECT_DOUBLE_EQ(result_h[0], 0.0073435040953447293);
  EXPECT_DOUBLE_EQ(result_h[1], 0.0041840526545260711);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution(result_x[i]), 1e-13);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_NEAR(result_h[i], result_x[i + 1] - result_x[i], 1e-15);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderIntegrateWithPIControl,
  IntegrateIntegratesWithStdValarrayForSystemOfEquations)
{
  constexpr double given_epsilon {1e-3};

  HigherOrderIntegrateWithPIControl integrate {
    CalculateNewY<DOPR853_s, decltype(example_f_with_std_valarray_2)>{
      example_f_with_std_valarray_2,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients},
    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>{
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      valarray<double>(given_epsilon, 2),
      valarray<double>(given_epsilon, 2)},
    ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};

  const auto result = integrate.integrate<2>(
    integrate_inputs_with_std_valarray_2);

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 1049);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.0032849410332814431);
  EXPECT_DOUBLE_EQ(result_x[2], 0.0051555348479529689);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 0.99681917261739073);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 1.0126176628239383);
  EXPECT_DOUBLE_EQ(result_h[0], 0.0032849410332814431);
  EXPECT_DOUBLE_EQ(result_h[1], 0.0018705938146715258);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution_2(result_x[i])[0], 1e-12);
    EXPECT_NEAR(result_y[i][1], exact_solution_2(result_x[i])[1], 1e-11);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_NEAR(result_h[i], result_x[i + 1] - result_x[i], 1e-15);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderIntegrateWithPIControl,
  IntegrateIntegratesWithNVectorForSystemOfEquations)
{
  constexpr double given_epsilon {1e-3};

  HigherOrderIntegrateWithPIControl integrate {
    CalculateNewY<DOPR853_s, decltype(example_f_with_NVector_2)>{
      example_f_with_NVector_2,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients},
    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, NVector<2>>{
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      NVector<2>(given_epsilon),
      NVector<2>(given_epsilon)},
    ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};

  const auto result = integrate.integrate<2>(integrate_inputs_with_nvector_2);

  const auto result_x = std::get<0>(result);
  const auto result_y = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  EXPECT_EQ(result_x.size(), 1049);
  EXPECT_DOUBLE_EQ(result_x[0], 0.0);
  EXPECT_DOUBLE_EQ(result_x[1], 0.0032849410332814431);
  EXPECT_DOUBLE_EQ(result_x[2], 0.0051555348479529689);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 2], 0.99681917261739073);
  EXPECT_DOUBLE_EQ(result_x[result_x.size() - 1], 1.0126176628239383);
  EXPECT_DOUBLE_EQ(result_h[0], 0.0032849410332814431);
  EXPECT_DOUBLE_EQ(result_h[1], 0.0018705938146715258);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_y[i][0], exact_solution_2(result_x[i])[0], 1e-12);
    EXPECT_NEAR(result_y[i][1], exact_solution_2(result_x[i])[1], 1e-11);
  }

  for (std::size_t i {0}; i < result_x.size() - 1; ++i)
  {
    EXPECT_NEAR(result_h[i], result_x[i + 1] - result_x[i], 1e-15);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestHigherOrderIntegrateWithPIControl,
  IntegrateIntegratesForcedOscillation)
{
  HigherOrderIntegrateWithPIControl integrate {
    CalculateNewY<
      DOPR853_s,
      decltype(forced_oscillation_eq_of_motion<valarray<double>>)
      >{
      forced_oscillation_eq_of_motion<valarray<double>>,
      DOPR853_a_coefficients,
      DOPR853_b_coefficients,
      DOPR853_c_coefficients},
    CalculateError<DOPR853_s, DOPR853_BHHCoefficientSize, valarray<double>>{
      DOPR853_delta_coefficients,
      DOPR853_bhh_coefficients,
      valarray<double>(epsilon, 1),
      valarray<double>(epsilon, 2)},
    ComputePIStepSize{1.0 / 8.0 - beta_8, beta_8, 0.333, 6.0}};

  IntegrationInputs<valarray<double>> inputs {
    valarray<double>{0.0, 0.0},
    0.0,
    110.0};

  const auto result = integrate.integrate<2>(inputs);

  const auto result_t = std::get<0>(result);
  const auto result_x = std::get<1>(result);
  const auto result_h = std::get<2>(result);

  ForcedOscillationExactSolution ex {};

  EXPECT_EQ(result_t.size(), 50001);
  EXPECT_EQ(result_x.size(), 50001);
  EXPECT_EQ(result_h.size(), 50000);

  for (std::size_t i {0}; i < result_x.size(); ++i)
  {
    EXPECT_NEAR(result_x[i][0], ex.compute_exact_solution(result_t[i]), 1e-8);
  }

  for (std::size_t i {0}; i < result_t.size() - 1; ++i)
  {
    EXPECT_NEAR(result_h[i], result_t[i + 1] - result_t[i], 1e-16);
  }
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests