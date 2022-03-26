#include "Numerical/ODE/Output.h"

#include "gtest/gtest.h"

#include <vector>

using Numerical::ODE::Output;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OutputTests, DefaultConstructs)
{
  Output out {};

  EXPECT_EQ(out.k_max_, 0);
  EXPECT_EQ(out.count_, 0);
  EXPECT_EQ(out.dense_, false);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OutputTests, ConstructsWithZeroValue)
{
  Output out {0};

  EXPECT_EQ(out.k_max_, 500);
  EXPECT_EQ(out.n_save_, 0);
  EXPECT_EQ(out.count_, 0);
  EXPECT_EQ(out.x_save_.size(), 500);
  EXPECT_EQ(out.dense_, false);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OutputTests, ConstructsWithNonZeroValue)
{
  Output out {50};

  EXPECT_EQ(out.k_max_, 500);
  EXPECT_EQ(out.n_save_, 50);
  EXPECT_EQ(out.count_, 0);
  EXPECT_EQ(out.x_save_.size(), 500);
  EXPECT_EQ(out.dense_, true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OutputTests, InitSetsValuesOnDefaultConstructedOutput)
{
  Output out {};
  out.init(2, 0.0, 1.0);
  EXPECT_EQ(out.n_var_, 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OutputTests, InitSetsValuesOnOutput)
{
  Output out {50};
  out.init(2, 0.0, 1.0);
  EXPECT_EQ(out.n_var_, 2);
  EXPECT_EQ(out.y_save_.size(), 500);
  EXPECT_EQ(out.x1_, 0.0);
  EXPECT_EQ(out.x2_, 1.0);
  EXPECT_EQ(out.x_out_, 0.0);
  EXPECT_EQ(out.dx_out_, 0.02);
}

} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests