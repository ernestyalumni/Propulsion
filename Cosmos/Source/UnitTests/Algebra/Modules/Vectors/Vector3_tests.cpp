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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, EqualityOperatorWorks)
{
  const Vector3<double> a {1.0, 2.0, 3.0};
  const Vector3<double> b {1.0, 2.0, 3.0};
  EXPECT_EQ(a, b);
  EXPECT_TRUE(a == b);

  EXPECT_EQ(b, a);
  EXPECT_TRUE(b == a);
  EXPECT_FALSE(a != b);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, InequalityOperatorWorks)
{
  const Vector3<double> a {1.0, 2.0, 3.0};
  const Vector3<double> b {4.0, 5.0, 6.0};
  EXPECT_NE(a, b);
  EXPECT_TRUE(a != b);

  EXPECT_NE(b, a);
  EXPECT_TRUE(b != a);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, VectorAdditionWorks)
{
  const Vector3<double> a {1.0, 2.0, 3.0};
  const Vector3<double> b {4.0, 5.0, 6.0};
  const Vector3<double> c {a + b};
  EXPECT_EQ(c.get_entry(0), 5.0);
  EXPECT_EQ(c.get_entry(1), 7.0);
  EXPECT_EQ(c.get_entry(2), 9.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Vector3Tests, VectorSubtractionWorks)
{
  const Vector3<double> a {1.0, 2.0, 3.0};
  const Vector3<double> b {4.0, 5.0, 6.0};
  const Vector3<double> c {a - b};
  EXPECT_EQ(c.x(), -3.0);
  EXPECT_EQ(c.y(), -3.0);
  EXPECT_EQ(c.z(), -3.0);
}

//------------------------------------------------------------------------------
// operator+= mutates *this in place and returns reference to *this.
//------------------------------------------------------------------------------
TEST(Vector3Tests, PlusEqualsMutatesInPlace)
{
  Vector3<double> a {1.0, 2.0, 3.0};
  const Vector3<double> b {4.0, 5.0, 6.0};
  Vector3<double>& ref {a += b};
  // a was mutated to the sum
  EXPECT_EQ(a.x(), 5.0);
  EXPECT_EQ(a.y(), 7.0);
  EXPECT_EQ(a.z(), 9.0);
  // return value is reference to a
  EXPECT_EQ(&ref, &a);
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests