#include "Algebra/Modules/Vectors/NVector.h"

#include "gtest/gtest.h"

#include <type_traits>

using Algebra::Modules::Vectors::NVector;

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
TEST(NVectorTests, DefaultConstructible)
{
  EXPECT_TRUE(std::is_default_constructible<NVector<2>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, CopyConstructible)
{
  EXPECT_TRUE(std::is_copy_constructible<NVector<2>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, CopyAssignable)
{
  EXPECT_TRUE(std::is_copy_assignable<NVector<2>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, MoveConstructible)
{
  EXPECT_TRUE(std::is_move_constructible<NVector<2>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, MoveAssignable)
{
  EXPECT_TRUE(std::is_move_assignable<NVector<2>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, Destructible)
{
  EXPECT_TRUE(std::is_destructible<NVector<2>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, HasVirtualDestructor)
{
  EXPECT_TRUE(std::has_virtual_destructor<NVector<2>>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, DefaultConstructsToZeroes)
{
  NVector<2> x {};
  EXPECT_EQ(x[0], 0);
  EXPECT_EQ(x[1], 0);
  EXPECT_EQ(x.dimension, 2);

  NVector<4> y {};
  EXPECT_EQ(y[0], 0);
  EXPECT_EQ(y[1], 0);
  EXPECT_EQ(y[2], 0);
  EXPECT_EQ(y[3], 0);
  EXPECT_EQ(y.dimension, 4);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, ConstructsWithInitializerList)
{
  NVector<3, double> A {1, 3, 6};
  EXPECT_EQ(A[0], 1);
  EXPECT_EQ(A[1], 3);
  EXPECT_EQ(A[2], 6);
  EXPECT_EQ(A.dimension, 3);
}

// cf. 12.4 Exercises, Exercise 1 of Apostol, Calculus, Vol. 1.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, BinaryAdditionWorks)
{
  const NVector<3> A {1, 3, 6};
  const NVector<3> B {4, -3, 3};
  const NVector<3> C {A + B};
  EXPECT_EQ(C[0], 5);
  EXPECT_EQ(C[1], 0);
  EXPECT_EQ(C[2], 9);
  EXPECT_EQ(C.dimension, 3);
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests