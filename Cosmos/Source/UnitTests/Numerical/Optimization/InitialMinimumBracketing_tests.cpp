#include "Numerical/Optimization/InitialMinimumBracketing.h"

#include "TestSetup.h"
#include "gtest/gtest.h"

#include <tuple>

using Numerical::Optimization::InitialMinimumBracketing;
using std::get;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace Optimization
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(InitialMinimumBracketTests, IsMinimumFoundFindsMinimumWithinRange)
{
  {
    const auto results = InitialMinimumBracketing::is_minimum_found(
      4.0,
      2.0,
      -1.0,
      example_f1(4.0),
      example_f1(2.0),
      example_f1(-1.0),
      0.0,
      -10.0,
      example_f1);

    EXPECT_TRUE(results.has_value());
    EXPECT_EQ((*results).at(0), 2.0);
    EXPECT_EQ((*results).at(1), 0.0);
    EXPECT_EQ((*results).at(2), -1.0);
  }
  {
    const auto results = InitialMinimumBracketing::is_minimum_found(
      -16.0,
      -15.0,
      5.0,
      example_f1(-16.0),
      example_f1(-15.0),
      example_f1(5.0),
      2.0,      
      6969.42,
      example_f1);

    EXPECT_TRUE(results.has_value());
    EXPECT_EQ((*results).at(0), -15.0);
    EXPECT_EQ((*results).at(1), 2.0);
    EXPECT_EQ((*results).at(2), 5.0);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(InitialMinimumBracketTests, IsMinimumFoundReturnsNullWhenNotFound)
{
  const auto results = InitialMinimumBracketing::is_minimum_found(
    0.0,
    4.0,
    25.0,
    example_f3(0.0),
    example_f3(4.0),
    example_f3(25.0),
    1.0500000000000001e+23,
    1000.0,
    example_f3);

  EXPECT_FALSE(results.has_value());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  InitialMinimumBracketTests,
  IsMinimumFoundReturnsNullWhenMinimumIsNotBounded)
{
  const auto results = InitialMinimumBracketing::is_minimum_found(
    -2442.0796,
    -1125.0,
    -311.0,
    example_f2(-2442.0796),
    example_f2(-1125.0),
    example_f2(-311.0),
    2.0,
    6969.42,
    example_f2);

  EXPECT_FALSE(results.has_value());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(InitialMinimumBracketTests, BracketMinimumFindsMinimumIfWithinRange)
{
  {
    const auto results = InitialMinimumBracketing::bracket_minimum(
      0.0,
      4.0,
      example_f2);

    EXPECT_TRUE(results.has_value());
    EXPECT_EQ((*results).at(0), 0.0);
    EXPECT_EQ((*results).at(1), 4.0);
    EXPECT_DOUBLE_EQ((*results).at(2), 10.472135999999999);
  }
  {
    const auto results = InitialMinimumBracketing::bracket_minimum(
      -2.0,
      5.0,
      example_f2);

    EXPECT_TRUE(results.has_value());
    EXPECT_EQ((*results).at(0), -2.0);
    EXPECT_EQ((*results).at(1), 5.0);
    EXPECT_DOUBLE_EQ((*results).at(2), 16.326238);
  }
  {
    const auto results = InitialMinimumBracketing::bracket_minimum(
      1.0,
      3.0,
      example_f2);

    EXPECT_TRUE(results.has_value());
    EXPECT_EQ((*results).at(0), 1.0);
    EXPECT_EQ((*results).at(1), 3.0);
    EXPECT_DOUBLE_EQ((*results).at(2), 6.2360679999999995);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(InitialMinimumBracketTests, BracketMinimumReturnsForOutOfRangeInput)
{
  {
    const auto results = InitialMinimumBracketing::bracket_minimum(
      111.0,
      125.0,
      example_f2);

    EXPECT_TRUE(results.has_value());
    EXPECT_DOUBLE_EQ((*results).at(0), 51.695047647815983);
    EXPECT_DOUBLE_EQ((*results).at(1), 1.9999999999999858);
    EXPECT_DOUBLE_EQ((*results).at(2), -7.6099052742137161);
  }
  {
    const auto results = InitialMinimumBracketing::bracket_minimum(
      -311.0,
      -1125.0,
      example_f2);

    EXPECT_TRUE(results.has_value());
    EXPECT_EQ((*results).at(0), -1125.0);
    EXPECT_EQ((*results).at(1), -311.0);
    EXPECT_DOUBLE_EQ((*results).at(2), 1006.0796760000001);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  InitialMinimumBracketTests,
  BracketMinimumReturnsForFunctionWithNoGlobalMinimum)
{
  {
    const auto results = InitialMinimumBracketing::bracket_minimum(
      0.0,
      4.0,
      example_f3);

    EXPECT_FALSE(results.has_value());
  }
  {
    const auto results = InitialMinimumBracketing::bracket_minimum(
      1.0,
      3.0,
      example_f3);

    EXPECT_FALSE(results.has_value());
  }
}

} // namespace Optimization
} // namespace Numerical
} // namespace GoogleUnitTests