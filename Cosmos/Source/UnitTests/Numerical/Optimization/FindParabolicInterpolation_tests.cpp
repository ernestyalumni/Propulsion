#include "Numerical/Optimization/FindParabolicInterpolation.h"

#include "TestSetup.h"
#include "gtest/gtest.h"

using Numerical::Optimization::FindParabolicInterpolation;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace Optimization
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FindParabolicInterpolationTests, FindExtremaOutsideOfGlobalExtrema)
{
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      42.0,
      69.0,
      112.68692,
      example_f1(42.0),
      example_f1(69.0),
      example_f1(112.68692));

    EXPECT_DOUBLE_EQ(results.extrema_position_, -7.1054273576010019e-15);
    EXPECT_EQ(results.second_derivative_, 1);
  }
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      69.0,
      0.001,
      -111.64173,
      example_f1(69.0),
      example_f1(0.001),
      example_f1(-111.64173));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 7.1054273576010019e-15);
    EXPECT_EQ(results.second_derivative_, 1);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FindParabolicInterpolationTests, FindExtremaFindsExtremaWhenWithinRange)
{
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      0.0,
      4.0,
      10.472136,
      example_f2(0.0),
      example_f2(4.0),
      example_f2(10.472136));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 2);
    EXPECT_EQ(results.second_derivative_, 1);
  }
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      -2.0,
      5.0,
      16.326239,
      example_f2(-2.0),
      example_f2(5.0),
      example_f2(16.326239));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 2);
    EXPECT_EQ(results.second_derivative_, 1);
  }
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      1.0,
      3.0,
      -2.236068,
      example_f2(1.0),
      example_f2(3.0),
      example_f2(-2.236068));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 2);
    EXPECT_EQ(results.second_derivative_, 1);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FindParabolicInterpolationTests, FindExtremaCalculatesOutOfRange)
{
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      111.0,
      125.0,
      147.65248,
      example_f2(111.0),
      example_f2(125.0),
      example_f2(147.65248));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 2.0000000000000426);
    EXPECT_DOUBLE_EQ(results.second_derivative_, 1);
  }
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      -311.0,
      -1125.0,
      -2442.0796,
      example_f2(-311.0),
      example_f2(-1125.0),
      example_f2(-2442.0796));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 1.9999999999998863);
    EXPECT_DOUBLE_EQ(results.second_derivative_, 1);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FindParabolicInterpolationTests, FindExtremaCalculatesForNoGlobalMinimum)
{
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      0.0,
      4.0,
      25.0,
      example_f3(0.0),
      example_f3(4.0),
      example_f3(25.0));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 1.0500000000000001e+23);
    EXPECT_DOUBLE_EQ(results.second_derivative_, 0);
  }
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      5.0,
      -2.0,
      -13.326238,
      example_f3(5.0),
      example_f3(-2.0),
      example_f3(-13.326238));

    EXPECT_DOUBLE_EQ(results.extrema_position_, -7.2648566631425414e+22);
    EXPECT_DOUBLE_EQ(results.second_derivative_, -0);
  }
  {
    const auto results = FindParabolicInterpolation::find_extrema(
      1.0,
      3.0,
      25.0,
      example_f3(1.0),
      example_f3(3.0),
      example_f3(25.0));

    EXPECT_DOUBLE_EQ(results.extrema_position_, 5.28e+22);
    EXPECT_DOUBLE_EQ(results.second_derivative_, 0);
  }
}

} // namespace Optimization
} // namespace Numerical
} // namespace GoogleUnitTests