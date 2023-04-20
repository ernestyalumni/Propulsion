#include "Numerical/Optimization/mins.h"

#include "gtest/gtest.h"

using Numerical::Optimization::Bracketmethod;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace Optimization
{

double example_f(const double x)
{
  return x * x + 1.0;
};

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

  bm.bracket(42.0, 69.0, example_f);

  SUCCEED();
}

} // namespace Optimization
} // namespace Numerical
} // namespace GoogleUnitTests