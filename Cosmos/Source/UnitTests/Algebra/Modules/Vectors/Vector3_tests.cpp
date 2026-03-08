#include "Algebra/Modules/Vectors/Vector3.h"

#include "gtest/gtest.h"

#include <type_traits>

using Algebra::Modules::Vectors::Vector3;

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
TEST(Vector3Tests, DefaultConstructible)
{
  EXPECT_TRUE(std::is_default_constructible<Vector3<>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, CopyConstructible)
{
  EXPECT_TRUE(std::is_copy_constructible<Vector3<>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, CopyAssignable)
{
  EXPECT_TRUE(std::is_copy_assignable<Vector3<>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, MoveConstructible)
{
  EXPECT_TRUE(std::is_move_constructible<Vector3<>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, MoveAssignable)
{
  EXPECT_TRUE(std::is_move_assignable<Vector3<>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, Destructible)
{
  EXPECT_TRUE(std::is_destructible<Vector3<>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, HasVirtualDestructor)
{
  EXPECT_TRUE(std::has_virtual_destructor<Vector3<>>());
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests