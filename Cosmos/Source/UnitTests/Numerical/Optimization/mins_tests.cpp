#include "Numerical/Optimization/mins.h"

#include "TestSetup.h"
#include "gtest/gtest.h"

#include <cmath> // std::isfinite
#include <limits> // std::numeric_limits

using Numerical::Optimization::Bracketmethod;
using Numerical::Optimization::GoldenSectionSearch;
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GoldenSectionSearchTests, DefaultConstructs)
{
  GoldenSectionSearch gss {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  GoldenSectionSearchTests,
  MinimizeFindsSmallerBracketGivenBracketWithMinimum)
{
  {
    GoldenSectionSearch gss {};

    const auto result = gss.minimize(example_f1, 69.0, 0.001, -111.64173);

    EXPECT_DOUBLE_EQ(std::get<0>(result), 1.0536712085491855e-08);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 1.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), 1.0536712396569381e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), 1.0536712204312896e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), 1.0536712085491855e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), 1.0536711893235373e-08);
  }
  {
    GoldenSectionSearch gss {};

    const auto result = gss.minimize(example_f1, 4.0, 2.0, -1.0);

    EXPECT_DOUBLE_EQ(std::get<0>(result), 1.0536712046017888e-08);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 1.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), 1.0536712288719647e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), 1.053671213872171e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), 1.0536712046017888e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), 1.0536711896019952e-08);
  }
  {
    GoldenSectionSearch gss {};

    const auto result = gss.minimize(example_f1, -16.0, -15.0, 5.0);

    EXPECT_DOUBLE_EQ(std::get<0>(result), -1.0536711866226056e-08);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 1.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), -1.0536712248186812e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), -1.0536712012122082e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), -1.0536711866226056e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), -1.0536711630161327e-08);
  }
  {
    GoldenSectionSearch gss {};

    const auto result = gss.minimize(example_f2, 3.0, 1.0, -2.236068);

    EXPECT_DOUBLE_EQ(std::get<0>(result), 2.0000000051394999);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 0.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), 2.0000000650461365);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), 2.0000000280217991);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), 2.0000000051394999);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), 1.9999999681151621);
  }
  {
    GoldenSectionSearch gss {};

    const auto result = gss.minimize(example_f2, -2.0, 5.0, 16.326238);

    EXPECT_DOUBLE_EQ(std::get<0>(result), 1.9999999960519785);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 0.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), 1.9999999465548441);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), 1.9999999771457555);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), 1.9999999960519785);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), 2.0000000266428901);
  }
  {
    GoldenSectionSearch gss {};

    const auto result = gss.minimize(example_f2, 1.0, 3.0, 6.2360679999999995);

    EXPECT_DOUBLE_EQ(std::get<0>(result), 1.9999999948604996);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 0.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), 1.9999999349538629);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), 1.9999999719782005);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), 1.9999999948604996);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), 2.0000000318848374);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GoldenSectionSearchTests, MinimizeDependsOnTolerance)
{
  {
    GoldenSectionSearch gss {1e-1};

    const auto result = gss.minimize(example_f1, 69.0, 0.001, -111.64173);

    EXPECT_DOUBLE_EQ(std::get<0>(result), 9.9367258006429412e-09);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 1.0);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), 1.0873177728278439e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), 1.0294418606998681e-08);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), 9.9367258006429412e-09);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), 9.3579666793631832e-09);
  }
  {
    GoldenSectionSearch gss {1e-1};

    const auto result = gss.minimize(example_f2, 3.0, 1.0, -2.236068);

    EXPECT_DOUBLE_EQ(std::get<0>(result), 2.0557280877260746);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 0.0031056197616052827);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(0), 2.2360679744093597);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(1), 2.124611794686341);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(2), 2.0557280877260746);
    EXPECT_DOUBLE_EQ(std::get<2>(result).at(3), 1.9442719080030562);
  }
}

} // namespace Optimization
} // namespace Numerical
} // namespace GoogleUnitTests