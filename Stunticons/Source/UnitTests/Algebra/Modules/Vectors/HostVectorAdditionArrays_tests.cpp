#include "Algebra/Modules/Vectors/HostVectorAdditionArrays.h"
#include "gtest/gtest.h"

#include <type_traits>

using Algebra::Modules::Vectors::HostVectorAdditionArrays;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostVectorAdditionArraysTests, DefaultConstructible)
{
  HostVectorAdditionArrays arrays {};

  EXPECT_EQ(arrays.number_of_elements_, 50000);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostVectorAdditionArraysTests, CanSetElements)
{
  HostVectorAdditionArrays arrays {};

  for (std::size_t i {0}; i < arrays.number_of_elements_; ++i)
  {
    arrays.h_A_[i] = static_cast<float>(i);
    arrays.h_B_[i] = static_cast<float>(i) + 2.0f;
  }

  for (std::size_t i {0}; i < arrays.number_of_elements_; ++i)
  {
    EXPECT_FLOAT_EQ(arrays.h_A_[i], static_cast<float>(i));
    EXPECT_FLOAT_EQ(arrays.h_B_[i], static_cast<float>(i) + 2.0f);
  }
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests