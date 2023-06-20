#include "Arithmetic/IntegerPower.h"
#include "gtest/gtest.h"

using Arithmetic::integer_power;

namespace GoogleUnitTests
{
namespace Arithmetic
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(IntegerPowerTests, CalculatesForPowersOfTwo)
{
  EXPECT_EQ(integer_power(2, 0), 1);
  EXPECT_EQ(integer_power(2, 1), 2);
  EXPECT_EQ(integer_power(2, 2), 4);
  EXPECT_EQ(integer_power(2, 3), 8);
  EXPECT_EQ(integer_power(2, 4), 16);
  EXPECT_EQ(integer_power(2, 5), 32);
}

} // namespace Arithmetic
} // namespace GoogleUnitTests