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
      if (output.size() == 0)
      {
        transform(
          y.begin(),
          y.end(),
          back_inserter(output),
          [x](const double y_value)
          {
            return y_value - x * x + 1.0;
          });
      }
      else
      {
        transform(
          y.begin(),
          y.end(),
          output.begin(),
          [x](const double y_value)
          {
            return y_value - x * x + 1.0;
          });        
      }
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, StepsForCalculateNewYOnRK4Coefficients)
{
  ExampleSetup setup {};
  EXPECT_EQ(setup.y_out_.size(), 1);
  EXPECT_EQ(setup.y_out_.at(0), 0.0);

  Examplef f {};
  // For RK4 method.
  KCoefficients<RK4_s, vector<double>> k_coefficients;
  k_coefficients.ith_coefficient(1) = setup.dydx_0_;

  TestCalculateNewYAndError<RK4_s, decltype(f)> new_y_and_err {
    f,
    RK4_a_coefficients,
    RK4_c_coefficients,
    RK4_b_coefficients};

  const double x_2 {setup.t_0_ + new_y_and_err.get_c_i(2) * setup.h_};
  EXPECT_EQ(x_2, 0.25);

  new_y_and_err.sum_a_and_k_products<std::vector<double>, 1>(
    k_coefficients,
    2,
    setup.h_,
    setup.y_out_);

  std::transform(
    setup.y_out_.begin(),
    setup.y_out_.end(),
    setup.y_0_.begin(),
    setup.y_out_.begin(),
    std::plus<double>());

  EXPECT_EQ(setup.y_out_.at(0), 0.875);

  f(x_2, setup.y_out_, k_coefficients.ith_coefficient(2));

  EXPECT_EQ(k_coefficients.get_ith_coefficient(2).at(0), 1.8125);

  // l = 3

  const double x_3 {setup.t_0_ + new_y_and_err.get_c_i(3) * setup.h_};

  new_y_and_err.sum_a_and_k_products<std::vector<double>, 1>(
    k_coefficients,
    3,
    setup.h_,
    setup.y_out_);

  std::transform(
    setup.y_out_.begin(),
    setup.y_out_.end(),
    setup.y_0_.begin(),
    setup.y_out_.begin(),
    std::plus<double>());

  EXPECT_EQ(setup.y_out_.at(0), 0.953125);

  f(x_3, setup.y_out_, k_coefficients.ith_coefficient(3));

  EXPECT_EQ(k_coefficients.get_ith_coefficient(3).at(0), 1.890625);

  // l = 3

  const double x_4 {setup.t_0_ + new_y_and_err.get_c_i(4) * setup.h_};

  new_y_and_err.sum_a_and_k_products<std::vector<double>, 1>(
    k_coefficients,
    4,
    setup.h_,
    setup.y_out_);

  std::transform(
    setup.y_out_.begin(),
    setup.y_out_.end(),
    setup.y_0_.begin(),
    setup.y_out_.begin(),
    std::plus<double>());

  EXPECT_EQ(setup.y_out_.at(0), 1.4453125);

  f(x_4, setup.y_out_, k_coefficients.ith_coefficient(4));

  EXPECT_EQ(k_coefficients.get_ith_coefficient(4).at(0), 2.1953125);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, CalculateNewYCalculatesKCoefficientsForRK4)
{
  ExampleSetup setup {};
  KCoefficients<RK4_s, vector<double>> k_coefficients;
  Examplef f {};

  CalculateNewYAndError<RK4_s, Examplef> new_y_and_err {
    f,
    RK4_a_coefficients,
    RK4_c_coefficients,
    RK4_b_coefficients};

  // Step 1

  new_y_and_err.calculate_new_y<vector<double>, 1>(
    setup.h_,
    setup.t_0_,
    setup.y_0_,
    setup.dydx_0_,
    k_coefficients,
    setup.y_out_);

  EXPECT_DOUBLE_EQ(k_coefficients.get_ith_coefficient(1).at(0), 1.5);
  EXPECT_DOUBLE_EQ(k_coefficients.get_ith_coefficient(2).at(0), 1.8125);
  EXPECT_DOUBLE_EQ(k_coefficients.get_ith_coefficient(3).at(0), 1.890625);
  EXPECT_DOUBLE_EQ(k_coefficients.get_ith_coefficient(4).at(0), 2.1953125);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 1.4453125);

  setup.y_out_ = setup.y_0_;

  for (std::size_t i {1}; i <= RK4_s; ++i)
  {
    setup.y_out_[0] += setup.h_ * (
      RK4_b_coefficients.get_ith_element(i) *
        k_coefficients.get_ith_coefficient(i).at(0));
  }

  EXPECT_DOUBLE_EQ(setup.y_out_[0], 1.4251302083333333);

  // Step 2

  vector<double> dydx {f(setup.t_0_ + setup.h_, setup.y_out_)};

  EXPECT_EQ(dydx.size(), 1);
  EXPECT_DOUBLE_EQ(dydx.at(0), 2.175130208333333);

  vector<double> y_in {setup.y_out_};  
  new_y_and_err.calculate_new_y<vector<double>, 1>(
    setup.h_,
    setup.t_0_ + setup.h_,
    y_in,
    dydx,
    k_coefficients,
    setup.y_out_);

  EXPECT_EQ(k_coefficients.get_ith_coefficient(1).size(), 1);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(1).at(0),
    2.175130208333334);
  EXPECT_EQ(k_coefficients.get_ith_coefficient(2).size(), 1);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(2).at(0),
    2.406412760416666);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(3).at(0),
    2.4642333984375);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(4).at(0),
    2.657246907552084);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 2.657246907552083);
  setup.y_out_ = y_in;

  for (std::size_t i {1}; i <= RK4_s; ++i)
  {
    setup.y_out_[0] += setup.h_ * (
      RK4_b_coefficients.get_ith_element(i) *
        k_coefficients.get_ith_coefficient(i).at(0));
  }

  EXPECT_DOUBLE_EQ(setup.y_out_[0], 2.639602661132812);

  // Step 3

  dydx = f(setup.t_0_ + 2 * setup.h_, setup.y_out_);

  EXPECT_EQ(dydx.size(), 1);
  EXPECT_DOUBLE_EQ(dydx.at(0), 2.6396026611328125);

  y_in = setup.y_out_;
  new_y_and_err.calculate_new_y<vector<double>, 1>(
    setup.h_,
    setup.t_0_ + 2 * setup.h_,
    y_in,
    dydx,
    k_coefficients,
    setup.y_out_);

  EXPECT_EQ(k_coefficients.get_ith_coefficient(1).size(), 1);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(1).at(0),
    2.639602661132812);
  EXPECT_EQ(k_coefficients.get_ith_coefficient(2).size(), 1);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(2).at(0),
    2.737003326416016);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(3).at(0),
    2.761353492736816);
  EXPECT_DOUBLE_EQ(
    k_coefficients.get_ith_coefficient(4).at(0),
    2.77027940750122);

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 4.0202794075012207);
  setup.y_out_ = y_in;

  for (std::size_t i {1}; i <= RK4_s; ++i)
  {
    setup.y_out_[0] += setup.h_ * (
      RK4_b_coefficients.get_ith_element(i) *
        k_coefficients.get_ith_coefficient(i).at(0));
  }

  EXPECT_DOUBLE_EQ(setup.y_out_[0], 4.006818970044454);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError,
  CalculateNewYCalculatesKCoefficientsForDormandPrince5Coefficients)
{
  ExampleSetup setup {};
  Examplef f {};

  CalculateNewYAndError new_y_and_err {
    f,
    DOPRI5_a_coefficients,
    DOPRI5_c_coefficients,
    DOPRI5_delta_coefficients};

  // Step 1.

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

  EXPECT_NEAR(setup.y_out_.at(0), exact_solution(setup.h_), 1e-5);

  // Step 2.

  vector<double> dydx {f(setup.t_0_ + setup.h_, setup.y_out_)};

  EXPECT_EQ(dydx.size(), 1);
  EXPECT_DOUBLE_EQ(dydx.at(0), 2.175644097222222);

  vector<double> y_in {setup.y_out_};  

  new_y_and_err.calculate_new_y<vector<double>, 1>(
    setup.h_,
    setup.t_0_ + setup.h_,
    y_in,
    dydx,
    setup.k_coefficients_,
    setup.y_out_);

  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(1).at(0),
    2.175644097222222);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(2).at(0),
    2.2832085069444443);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(3).at(0),
    2.341591707899305);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(4).at(0),
    2.5801328124999987);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(5).at(0),
    2.633202642222983);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(6).at(0),
    2.667628162582859);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(7).at(0),
    2.6408707492856616);

  EXPECT_NEAR(setup.y_out_.at(0), exact_solution(2 * setup.h_), 1e-4);

  // Step 3.

  dydx = f(setup.t_0_ + 2 * setup.h_, setup.y_out_);

  EXPECT_EQ(dydx.size(), 1);
  EXPECT_DOUBLE_EQ(dydx.at(0), 2.6408707492856616);

  y_in = setup.y_out_;

  new_y_and_err.calculate_new_y<vector<double>, 1>(
    setup.h_,
    setup.t_0_ + 2 * setup.h_,
    y_in,
    dydx,
    setup.k_coefficients_,
    setup.y_out_);

  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(1).at(0),
    2.6408707492856616);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(2).at(0),
    2.694957824214227);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(3).at(0),
    2.7205861576079746);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(4).at(0),
    2.77797279059516);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(5).at(0),
    2.786162739074309);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(6).at(0),
    2.7745870563781194);
  EXPECT_DOUBLE_EQ(
    setup.k_coefficients_.get_ith_coefficient(7).at(0),
    2.7591771182645264);

  EXPECT_NEAR(setup.y_out_.at(0), exact_solution(3 * setup.h_), 1e-4);
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

  EXPECT_DOUBLE_EQ(setup.y_out_.at(0), 2.6408707492856616);
  EXPECT_DOUBLE_EQ(setup.y_err_.at(0), -1.7718829684792992e-05);
  EXPECT_NEAR(setup.y_out_.at(0), exact_solution(setup.h_ + setup.h_), 1e-4);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests