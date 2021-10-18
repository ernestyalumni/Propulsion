#include "Numerical/ODE/RKMethods/RungeKuttaMethod.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>

#include "gtest/gtest.h"

using Numerical::ODE::RKMethods::CalculateNextStep;

constexpr std::initializer_list<double> rk_4_alphas {0.5, 0.5, 1.};
constexpr std::initializer_list<double> rk_4_betas {0.5, 0., 0.5, 0., 0., 1.};
constexpr std::initializer_list<double> rk_4_cs {1./6., 1./3., 1./3., 1./6.};

template <std::size_t M>
class TestCalculateNextStep : public CalculateNextStep<M>
{
  public:

    using CalculateNextStep<M>::CalculateNextStep;

    using CalculateNextStep<M>::calculate_k_coefficients;
    using CalculateNextStep<M>::sum_beta_and_k_products;    
};

template <typename ContainerT, typename Field = double>
class ExponentialDecay
{
  public:

    ExponentialDecay() = default;

    ContainerT operator()(const Field, const ContainerT& x_n)
    {
      ContainerT result;

      std::transform(
        x_n.begin(),
        x_n.end(),
        result.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          -0.5));

      return result;
    }
};

constexpr std::array<double, 3> t_n {0., 0.11487653, 1.26364188};

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
TEST(TestCalculateNextStep, ConstructsFromInitializerLists)
{
  const CalculateNextStep<4> rk4 {
    {0.5, 0.5, 1.},
    {0.5, 0., 0.5, 0., 0., 1.},
    {1./6., 1./3., 1./3., 1./6.}};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNextStep, GetBetaIJGetsValues)
{
  const CalculateNextStep<4> rk4 {rk_4_alphas, rk_4_betas, rk_4_cs};

  EXPECT_EQ(rk4.get_beta_ij(2, 1), 0.5);
  EXPECT_EQ(rk4.get_beta_ij(3, 1), 0.0);
  EXPECT_EQ(rk4.get_beta_ij(3, 2), 0.5);
  EXPECT_EQ(rk4.get_beta_ij(4, 1), 0.0);
  EXPECT_EQ(rk4.get_beta_ij(4, 2), 0.0);
  EXPECT_EQ(rk4.get_beta_ij(4, 3), 1.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNextStep, GetAlphaGetsValues)
{
  const CalculateNextStep<4> rk4 {rk_4_alphas, rk_4_betas, rk_4_cs};

  EXPECT_EQ(rk4.get_alpha_i(2), 0.5);
  EXPECT_EQ(rk4.get_alpha_i(3), 0.5);
  EXPECT_EQ(rk4.get_alpha_i(4), 1.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNextStep, GetCIGetsValues)
{
  const CalculateNextStep<4> rk4 {rk_4_alphas, rk_4_betas, rk_4_cs};

  EXPECT_EQ(rk4.get_c_i(1), 1./6.);
  EXPECT_EQ(rk4.get_c_i(2), 1./3.);
  EXPECT_EQ(rk4.get_c_i(3), 1./3.);
  EXPECT_EQ(rk4.get_c_i(4), 1./6.);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNextStep, CalculateKCoefficientsComputesWithArrays)
{
  constexpr double acceptable_error {1e-8};

  std::array<double, 3> x_n {2., 4., 8.};

  TestCalculateNextStep<4> rk4 {rk_4_alphas, rk_4_betas, rk_4_cs};

  const auto result = rk4.calculate_k_coefficients<
    std::array<double, 3>,
    ExponentialDecay<std::array<double, 3>>
    >(x_n, t_n[0], t_n[1] - t_n[0]);

  EXPECT_EQ(result[0][0], -1.0);
  EXPECT_EQ(result[0][1], -2.0);
  EXPECT_EQ(result[0][2], -4.0);
  EXPECT_NEAR(result[1][0], -0.97128087, acceptable_error);
  EXPECT_NEAR(result[1][1], -1.94256173, acceptable_error);
  EXPECT_DOUBLE_EQ(result[1][2], -3.88512347);
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests