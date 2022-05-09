#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/calculate_hermite_interpolation.h"
#include "gtest/gtest.h"

#include <valarray>

using Numerical::ODE::RKMethods::calculate_hermite_interpolation;
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

struct ExampleSetupWithStdValarray
{
  double h_ {0.5};
  const valarray<double> y_0_ {0.5};
  const valarray<double> dydx_0_ {1.5};
  const valarray<double> y_1_ {1.4251302083333333};
  const valarray<double> dydx_1_ {2.175644097222222};
  const valarray<double> y_2_ {2.657246907552083};
  const valarray<double> dydx_2_ {2.6408707492856616};
  const valarray<double> y_3_ {4.0202794075012207};
  const valarray<double> dydx_3_ {2.7591771182645264};
};

struct ExampleSetupWithNVector
{
  double h_ {0.5};
  const NVector<1> y_0_ {0.5};
  const NVector<1> dydx_0_ {1.5};
  const NVector<1> y_1_ {1.4251302083333333};
  const NVector<1> dydx_1_ {2.175644097222222};
  const NVector<1> y_2_ {2.657246907552083};
  const NVector<1> dydx_2_ {2.6408707492856616};
  const NVector<1> y_3_ {4.0202794075012207};
  const NVector<1> dydx_3_ {2.7591771182645264};
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateHermiteInterpolation, InterpolatesMidwayWithStdValarray)
{
  ExampleSetupWithStdValarray setup {};

  {
    const auto result = calculate_hermite_interpolation(
      setup.y_0_,
      setup.y_1_,
      setup.dydx_0_,
      setup.dydx_1_,
      0.5,
      setup.h_);
    EXPECT_EQ(result[0], 0.9203373480902778);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_1_,
      setup.y_2_,
      setup.dydx_1_,
      setup.dydx_2_,
      0.5,
      setup.h_);
    EXPECT_EQ(result[0], 2.012111892188743);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_2_,
      setup.y_3_,
      setup.dydx_2_,
      setup.dydx_3_,
      0.5,
      setup.h_);
    EXPECT_EQ(result[0], 3.331369009465473);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateHermiteInterpolation, InterpolatesMidwayWithNVector)
{
  ExampleSetupWithNVector setup {};

  {
    const auto result = calculate_hermite_interpolation(
      setup.y_0_,
      setup.y_1_,
      setup.dydx_0_,
      setup.dydx_1_,
      0.5,
      setup.h_);
    EXPECT_EQ(result[0], 0.9203373480902778);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_1_,
      setup.y_2_,
      setup.dydx_1_,
      setup.dydx_2_,
      0.5,
      setup.h_);
    EXPECT_EQ(result[0], 2.012111892188743);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_2_,
      setup.y_3_,
      setup.dydx_2_,
      setup.dydx_3_,
      0.5,
      setup.h_);
    EXPECT_EQ(result[0], 3.331369009465473);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateHermiteInterpolation, WorksForTheta0WithStdValarray)
{
  ExampleSetupWithStdValarray setup {};

  {
    const auto result = calculate_hermite_interpolation(
      setup.y_0_,
      setup.y_1_,
      setup.dydx_0_,
      setup.dydx_1_,
      0.0,
      setup.h_);
    EXPECT_EQ(result[0], 0.5);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_1_,
      setup.y_2_,
      setup.dydx_1_,
      setup.dydx_2_,
      0.0,
      setup.h_);
    EXPECT_EQ(result[0], 1.4251302083333333);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_2_,
      setup.y_3_,
      setup.dydx_2_,
      setup.dydx_3_,
      0.0,
      setup.h_);
    EXPECT_EQ(result[0], 2.657246907552083);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateHermiteInterpolation, WorksForTheta0WithNVector)
{
  ExampleSetupWithNVector setup {};

  {
    const auto result = calculate_hermite_interpolation(
      setup.y_0_,
      setup.y_1_,
      setup.dydx_0_,
      setup.dydx_1_,
      0.0,
      setup.h_);
    EXPECT_EQ(result[0], 0.5);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_1_,
      setup.y_2_,
      setup.dydx_1_,
      setup.dydx_2_,
      0.0,
      setup.h_);
    EXPECT_EQ(result[0], 1.4251302083333333);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_2_,
      setup.y_3_,
      setup.dydx_2_,
      setup.dydx_3_,
      0.0,
      setup.h_);
    EXPECT_EQ(result[0], 2.657246907552083);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateHermiteInterpolation, WorksForTheta1WithStdValarray)
{
  ExampleSetupWithStdValarray setup {};

  {
    const auto result = calculate_hermite_interpolation(
      setup.y_0_,
      setup.y_1_,
      setup.dydx_0_,
      setup.dydx_1_,
      1.0,
      setup.h_);
    EXPECT_EQ(result[0], 1.4251302083333333);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_1_,
      setup.y_2_,
      setup.dydx_1_,
      setup.dydx_2_,
      1.0,
      setup.h_);
    EXPECT_EQ(result[0], 2.657246907552083);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_2_,
      setup.y_3_,
      setup.dydx_2_,
      setup.dydx_3_,
      1.0,
      setup.h_);
    EXPECT_EQ(result[0], 4.0202794075012207);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateHermiteInterpolation, WorksForTheta1WithNVector)
{
  ExampleSetupWithNVector setup {};

  {
    const auto result = calculate_hermite_interpolation(
      setup.y_0_,
      setup.y_1_,
      setup.dydx_0_,
      setup.dydx_1_,
      1.0,
      setup.h_);
    EXPECT_EQ(result[0], 1.4251302083333333);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_1_,
      setup.y_2_,
      setup.dydx_1_,
      setup.dydx_2_,
      1.0,
      setup.h_);
    EXPECT_EQ(result[0], 2.657246907552083);
  }
  {
    const auto result = calculate_hermite_interpolation(
      setup.y_2_,
      setup.y_3_,
      setup.dydx_2_,
      setup.dydx_3_,
      1.0,
      setup.h_);
    EXPECT_EQ(result[0], 4.0202794075012207);
  }
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests