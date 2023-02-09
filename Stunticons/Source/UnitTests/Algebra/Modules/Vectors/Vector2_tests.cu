#include "gtest/gtest.h"

#include <type_traits>

//using Algebra::Modules::Vectors::NVector;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Vectors
{

constexpr double epsilon {1e-6};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector2Tests, DefaultConstructible)
{
  //EXPECT_TRUE(std::is_default_constructible<NVector<2>>());
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests