#include "Numerical/Optimization/mins.h"

#include "TestSetup.h"
#include "gtest/gtest.h"

#include <cmath> // std::isfinite
#include <limits> // std::numeric_limits

using Numerical::Optimization::Bracketmethod;
using std::isfinite;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace Optimization
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BracketmethodTests, DefaultConstructs)
{
  Bracketmethod bm {};
  EXPECT_EQ(bm.ax, 0);
  EXPECT_EQ(bm.bx, 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BracketmethodTests, BracketMethodMutatesDataMembers)
{
  Bracketmethod bm {};

  bm.bracket(42.0, 69.0, example_f1);

  EXPECT_EQ(bm.ax, 42.0);
  EXPECT_EQ(bm.bx, 69.0);
  EXPECT_FLOAT_EQ(bm.cx, 112.68692);

  bm.bracket(0.001, 69.0, example_f1);

  EXPECT_EQ(bm.ax, 69.0);
  EXPECT_EQ(bm.bx, 0.001);
  EXPECT_FLOAT_EQ(bm.cx, -111.64173);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BracketmethodTests, BracketMethodFindsMinimumIfWithinRange)
{
  Bracketmethod bm {};

  bm.bracket(0.0, 4.0, example_f2);

  EXPECT_EQ(bm.ax, 0.0);
  EXPECT_EQ(bm.bx, 4.0);
  EXPECT_FLOAT_EQ(bm.cx, 10.472136);
  EXPECT_EQ(bm.fa, 4.0);
  EXPECT_EQ(bm.fb, 4.0);
  EXPECT_FLOAT_EQ(bm.fc, 71.7771);

  bm.bracket(-2.0, 5.0, example_f2);
  EXPECT_EQ(bm.ax, -2.0);
  EXPECT_EQ(bm.bx, 5.0);
  EXPECT_FLOAT_EQ(bm.cx, 16.326239);
  EXPECT_EQ(bm.fa, 16.0);
  EXPECT_EQ(bm.fb, 4.0);
  EXPECT_FLOAT_EQ(bm.fc, 205.24109);

  bm.bracket(1.0, 3.0, example_f2);
  EXPECT_EQ(bm.ax, 3.0);
  EXPECT_EQ(bm.bx, 1.0);
  EXPECT_FLOAT_EQ(bm.cx, -2.236068);
  EXPECT_EQ(bm.fa, 4);
  EXPECT_EQ(bm.fb, 1);
  EXPECT_FLOAT_EQ(bm.fc, 17.944273);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BracketmethodTests, BracketMethodReturnsResultsForOutOfRangeInput)
{
  {
    Bracketmethod bm {};
    bm.bracket(111.0, 125.0, example_f2);
    EXPECT_EQ(bm.ax, 111.0);
    EXPECT_EQ(bm.bx, 125.0);
    EXPECT_FLOAT_EQ(bm.cx, 147.65248);
    EXPECT_EQ(bm.fa, 11881);
    EXPECT_EQ(bm.fb, 4.0);
    EXPECT_FLOAT_EQ(bm.fc, 21214.645);
  }
  {
    Bracketmethod bm {};
    bm.bracket(-311.0, -1125.0, example_f2);
    EXPECT_EQ(bm.ax, -311);
    EXPECT_EQ(bm.bx, -1125);
    EXPECT_FLOAT_EQ(bm.cx, -2442.0796);
    EXPECT_EQ(bm.fa, 97969);
    EXPECT_EQ(bm.fb, 4.0);
    EXPECT_FLOAT_EQ(bm.fc, 5973525.5);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BracketmethodTests, BracketMethodResultsForFunctionWithNoGlobalMinimum)
{
  Bracketmethod bm {};

  bm.bracket(0.0, 4.0, example_f3);

  EXPECT_FLOAT_EQ(bm.ax, 1.70558e+308);
  EXPECT_FALSE(isfinite(bm.bx));
  EXPECT_FALSE(isfinite(bm.cx));
  EXPECT_FLOAT_EQ(bm.fa, -1.70558e+308);
  EXPECT_FALSE(isfinite(bm.fb));
  EXPECT_FALSE(isfinite(bm.fc));

  bm.bracket(-2.0, 5.0, example_f3);
  EXPECT_EQ(bm.ax, 5.0);
  EXPECT_EQ(bm.bx, -2.0);
  EXPECT_FLOAT_EQ(bm.cx, -13.326238);
  EXPECT_FALSE(isfinite(bm.fa));
  EXPECT_EQ(bm.fb, 2.0);
  EXPECT_FLOAT_EQ(bm.fc, 13.326238);

  bm.bracket(1.0, 3.0, example_f3);
  EXPECT_FLOAT_EQ(bm.ax, 1.66159e+308);
  EXPECT_FALSE(isfinite(bm.bx));
  EXPECT_FALSE(isfinite(bm.cx));
  EXPECT_FLOAT_EQ(bm.fa, -1.66159e+308);
  EXPECT_FALSE(isfinite(bm.fb));
  EXPECT_FALSE(isfinite(bm.fc));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BracketmethodTests, BracketMethodResultsDependUponInitialConstruction)
{
  {
    Bracketmethod bm {};
    bm.bracket(-2.0, 5.0, example_f3);
    EXPECT_FLOAT_EQ(bm.ax, 1.14008e+308);
    EXPECT_FALSE(isfinite(bm.bx));
    EXPECT_FALSE(isfinite(bm.cx));
    EXPECT_FLOAT_EQ(bm.fa, -1.0 * std::numeric_limits<double>::infinity());
    EXPECT_FALSE(isfinite(bm.fb));
    EXPECT_FALSE(isfinite(bm.fc));
  }
  {
    Bracketmethod bm {};
    bm.bracket(1.0, 3.0, example_f3);
    EXPECT_EQ(bm.ax, 3.0);
    EXPECT_EQ(bm.bx, 1.0);
    EXPECT_FLOAT_EQ(bm.cx, -2.236068);
    EXPECT_EQ(bm.fa, -0);
    EXPECT_EQ(bm.fb, -1);
    EXPECT_FLOAT_EQ(bm.fc, 2.236068);
  }
}

} // namespace Optimization
} // namespace Numerical
} // namespace GoogleUnitTests