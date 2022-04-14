#include "Numerical/ODE/RKMethods/CalculateNewYAndError.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "Numerical/ODE/RKMethods/Coefficients/KCoefficients.h"
#include "Numerical/ODE/RKMethods/Coefficients/RK4Coefficients.h"

#include "gtest/gtest.h"

#include <algorithm> // std::transform
#include <array>
#include <cmath>
#include <iterator>
#include <vector>

using Numerical::ODE::RKMethods::Coefficients::KCoefficients;

using Numerical::ODE::RKMethods::CalculateNewYAndError;
using std::back_inserter;
using std::transform;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

constexpr std::size_t RK4_s {::Numerical::ODE::RKMethods::RK4Coefficients::s};

const auto& RK4_a_coefficients =
  ::Numerical::ODE::RKMethods::RK4Coefficients::a_coefficients;

const auto& RK4_b_coefficients =
  ::Numerical::ODE::RKMethods::RK4Coefficients::b_coefficients;

const auto& RK4_c_coefficients =
  ::Numerical::ODE::RKMethods::RK4Coefficients::c_coefficients;

constexpr std::size_t DOPRI5_s {
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::s};

const auto& DOPRI5_a_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::a_coefficients;

const auto& DOPRI5_c_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::c_coefficients;

const auto& DOPRI5_delta_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::delta_coefficients;

template <std::size_t S, typename DerivativeType, typename Field = double>
class TestCalculateNewYAndError :
  public CalculateNewYAndError<S, DerivativeType, Field>
{
  public:

    using CalculateNewYAndError<S, DerivativeType, Field>::
      CalculateNewYAndError;

    using CalculateNewYAndError<S, DerivativeType, Field>::
      sum_a_and_k_products;

    using CalculateNewYAndError<S, DerivativeType, Field>::get_a_ij;

    using CalculateNewYAndError<S, DerivativeType, Field>::get_c_i;
};

class Examplef
{
  public:

    Examplef() = default;

    void operator()(
      const double x,
      const vector<double>& y,
      vector<double>& output)
    {
      transform(
        y.begin(),
        y.end(),
        back_inserter(output),
        [x](const double y_value)
        {
          return y_value - x * x + 1.0;
        }
      ); 
    }

    vector<double> operator()(
      const double x,
      const vector<double>& y)
    {
      vector<double> output;
      this->operator()(x, y, output);
      return output;
    }
};

auto examplef = [](
  const double x,
  const vector<double>& y,
  vector<double>& output)
{
  transform(
    y.begin(),
    y.end(),
    back_inserter(output),
    [x](const double y_value)
    {
      return y_value - x * x + 1.0;
    });
};

auto exact_solution = [](const double x)
{
  return x * x + 2.0 * x + 1.0 - 0.5 * std::exp(x);
};

struct ExampleSetup
{
  double h_ {0.5};
  const double t_0_ {0.0};
  const double t_f_ {2.0};
  const vector<double> y_0_ {0.5};
  const vector<double> dydx_0_ {1.5};

  vector<double> y_out_ {0.0};
  vector<double> y_err_ {0.0};

  KCoefficients<DOPRI5_s, vector<double>> k_coefficients_;

  ExampleSetup()
  {
    k_coefficients_[0] = dydx_0_;
  }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, ConstructsFromRValueDerivative)
{
  {
    CalculateNewYAndError new_y_and_err {
      Examplef{},
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, Examplef> new_y_and_err {
      Examplef{},
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, ConstructsFromLValueDerivative)
{
  Examplef f {};
  {
    CalculateNewYAndError new_y_and_err {
      f,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, Examplef> new_y_and_err {
      f,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, decltype(f)> new_y_and_err {
      f,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError new_y_and_err {
      examplef,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, decltype(examplef)> new_y_and_err {
      examplef,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, ConstructsWithRK4Coefficients)
{
  Examplef f {};

  TestCalculateNewYAndError<RK4_s, decltype(f)> new_y_and_err {
    f,
    RK4_a_coefficients,
    RK4_c_coefficients,
    RK4_b_coefficients};

    EXPECT_EQ(new_y_and_err.get_a_ij(2, 1), 0.5);
    EXPECT_EQ(new_y_and_err.get_a_ij(3, 1), 0.0);
    EXPECT_EQ(new_y_and_err.get_a_ij(3, 2), 0.5);
    EXPECT_EQ(new_y_and_err.get_a_ij(4, 1), 0.0);
    EXPECT_EQ(new_y_and_err.get_a_ij(4, 2), 0.0);
    EXPECT_EQ(new_y_and_err.get_a_ij(4, 3), 1.0);

    EXPECT_EQ(new_y_and_err.get_c_i(2), 0.5);
    EXPECT_EQ(new_y_and_err.get_c_i(3), 0.5);
    EXPECT_EQ(new_y_and_err.get_c_i(4), 1.0);
}

//------------------------------------------------------------------------------
// Demonstrate the steps within the class member function sum_a_and_k_products
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, StepsForSumAAndKProductsWorksOnRK4)
{
  ExampleSetup setup {};
  Examplef f {};
  TestCalculateNewYAndError<RK4_s, decltype(f)> new_y_and_err {
    f,
    RK4_a_coefficients,
    RK4_c_coefficients,
    RK4_b_coefficients};

  // k coefficients behave nominally.
  KCoefficients<RK4_s, vector<double>> k_coefficients;
  EXPECT_EQ(k_coefficients.size(), 4);
  k_coefficients.ith_coefficient(1) = setup.dydx_0_;
  EXPECT_EQ(k_coefficients.get_ith_coefficient(1).at(0), 1.5);
  EXPECT_EQ(k_coefficients.get_ith_coefficient(2).size(), 0);

  std::array<double, 1> a_lj_times_k_j {};

  vector<double> out (1);
  out.at(0) = 0.0;

  k_coefficients.scalar_multiply(
    a_lj_times_k_j, 1,
    new_y_and_err.get_a_ij(2, 1));

  EXPECT_EQ(a_lj_times_k_j[0], 0.75);

  std::copy(
    a_lj_times_k_j.begin(),
    a_lj_times_k_j.end(),
    out.begin());

  EXPECT_EQ(out.at(0), 0.75);

  std::transform(
    out.begin(),
    out.end(),
    out.begin(),
    std::bind(
      std::multiplies<double>(),
      std::placeholders::_1,
      setup.h_));

  EXPECT_EQ(out.size(), 1);
  EXPECT_EQ(out.at(0), 0.375);

  // l = 3 case.

  k_coefficients.ith_coefficient(2) = vector<double>{1.8125};
  EXPECT_EQ(k_coefficients.get_ith_coefficient(2).at(0), 1.8125);
  EXPECT_EQ(k_coefficients.get_ith_coefficient(3).size(), 0);

  k_coefficients.scalar_multiply(
    a_lj_times_k_j, 1,
    new_y_and_err.get_a_ij(3, 1));

  EXPECT_EQ(a_lj_times_k_j[0], 0.0);

  std::copy(a_lj_times_k_j.begin(), a_lj_times_k_j.end(), out.begin());

  EXPECT_EQ(out.at(0), 0.0);

  // l = 3, j = 2 case.

  k_coefficients.scalar_multiply(
    a_lj_times_k_j,
    2,
    new_y_and_err.get_a_ij(3, 2));

  std::transform(
    out.begin(),
    out.end(),
    a_lj_times_k_j.begin(),
    out.begin(),
    std::plus<double>());

  EXPECT_EQ(out.at(0), 0.90625);

  std::transform(
    out.begin(),
    out.end(),
    out.begin(),
    std::bind(std::multiplies<double>(),
    std::placeholders::_1, setup.h_));

  EXPECT_EQ(out.at(0), 0.453125);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, SumAAndKProductsWorks)
{
  ExampleSetup setup {};
  Examplef f {};

  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(1).at(0), 1.5);
  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(2).size(), 0);
  EXPECT_EQ(setup.y_out_.size(), 1);
  EXPECT_EQ(setup.y_out_.at(0), 0.0);

  TestCalculateNewYAndError<DOPRI5_s, decltype(f)> new_y_and_err {
    f,
    DOPRI5_a_coefficients,
    DOPRI5_c_coefficients,
    DOPRI5_delta_coefficients};

  new_y_and_err.sum_a_and_k_products<std::vector<double>, 1>(
    setup.k_coefficients_,
    2,
    setup.h_,
    setup.y_out_);

  EXPECT_EQ(setup.k_coefficients_.get_ith_coefficient(1).at(0), 1.5);
  EXPECT_EQ(setup.k_coefficients_.get_ith_coefficient(2).size(), 0);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 0.15);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, SumAAndKProductsWorksForRK4Method)
{
  ExampleSetup setup {};
  Examplef f {};
  // For RK4 method.
  KCoefficients<RK4_s, vector<double>> k_coefficients;
  k_coefficients.ith_coefficient(1) = setup.dydx_0_;

  EXPECT_DOUBLE_EQ(k_coefficients.get_ith_coefficient(1).at(0), 1.5);
  EXPECT_DOUBLE_EQ(k_coefficients.get_ith_coefficient(2).size(), 0);
  EXPECT_EQ(setup.y_out_.size(), 1);
  EXPECT_EQ(setup.y_out_.at(0), 0.0);

  TestCalculateNewYAndError<RK4_s, decltype(f)> new_y_and_err {
    f,
    RK4_a_coefficients,
    RK4_c_coefficients,
    RK4_b_coefficients};

  new_y_and_err.sum_a_and_k_products<std::vector<double>, 1>(
    k_coefficients,
    2,
    setup.h_,
    setup.y_out_);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 0.375);

  // l = 3 case

  k_coefficients.ith_coefficient(2) = vector<double>{1.8125};
  EXPECT_EQ(k_coefficients.get_ith_coefficient(2).at(0), 1.8125);
  EXPECT_EQ(k_coefficients.get_ith_coefficient(3).size(), 0);

  new_y_and_err.sum_a_and_k_products<std::vector<double>, 1>(
    k_coefficients,
    3,
    setup.h_,
    setup.y_out_);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 0.453125);
}

/*
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, CalculateNewYCalculatesKCoefficients)
{
  ExampleSetup setup {};

  CalculateNewYAndError new_y_and_err {
    Examplef{},
    DOPRI5_a_coefficients,
    DOPRI5_c_coefficients,
    DOPRI5_delta_coefficients};

  new_y_and_err.calculate_new_y<vector<double>, 1>(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_,
    setup.y_out_);

  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(1).at(0), 1.5);
  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(2).at(0), 1.64);
  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(3).at(0), 1.71825);
  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(4).at(0),
    2.066666666666666);
  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(5).at(0),
    2.1469574759945127);
  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(6).at(0),
    2.2092840909090903);
  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(7).at(0),
    2.175644097222222);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 1.425644097222222);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, CalculatesNewYAndError)
{
  ExampleSetup setup {};

  CalculateNewYAndError new_y_and_err {
    Examplef{},
    DOPRI5_a_coefficients,
    DOPRI5_c_coefficients,
    DOPRI5_delta_coefficients};

  new_y_and_err.calculate_new_y_and_error<vector<double>, 1>(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    setup.k_coefficients_,
    setup.y_out_,
    setup.y_err_);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 1.425644097222222);
  EXPECT_DOUBLE_EQ(setup.y_err_.at(0), -2.4370659722241367e-05);
  EXPECT_NEAR(setup.y_out_.at(0), exact_solution(setup.h_), 1e-5);

  new_y_and_err.calculate_new_y_and_error<vector<double>, 1>(
    setup.h_,
    setup.t_0_ + setup.h_,
    vector<double>{setup.y_out_},
    vector<double>{setup.k_coefficients_.get_ith_coefficient(DOPRI5_s)},
    setup.k_coefficients_,
    setup.y_out_,
    setup.y_err_);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 2.3820792665834776);
  EXPECT_DOUBLE_EQ(setup.y_err_.at(0), 0.00039204193491992542);
  EXPECT_NEAR(setup.y_out_.at(0), exact_solution(setup.h_ + setup.h_), 1e-5);
}
*/
} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests