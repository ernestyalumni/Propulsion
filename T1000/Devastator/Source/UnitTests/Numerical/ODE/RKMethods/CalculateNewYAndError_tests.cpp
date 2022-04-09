#include "Numerical/ODE/RKMethods/CalculateNewYAndError.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "Numerical/ODE/RKMethods/Coefficients/KCoefficients.h"

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
TEST(TestCalculateNewYAndError, SumAAndKProductsWorks)
{
  ExampleSetup setup {};
  Examplef f {};

  EXPECT_DOUBLE_EQ(setup.k_coefficients_.get_ith_coefficient(1).at(0), 1.5);

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

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 0.15);
}

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

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests