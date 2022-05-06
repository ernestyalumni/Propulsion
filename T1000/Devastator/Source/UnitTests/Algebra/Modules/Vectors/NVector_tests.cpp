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

const NVector<3> B {4, -3, 3};
const NVector<3> C {2, 1, 5};

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
  const NVector<3> result {A + B};
  EXPECT_EQ(result[0], 5);
  EXPECT_EQ(result[1], 0);
  EXPECT_EQ(result[2], 9);
  EXPECT_EQ(result.dimension, 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, BinarySubtractionWorks)
{
  const NVector<3> A {1, 3, 6};
  const NVector<3> result {A - B};
  EXPECT_EQ(result[0], -3);
  EXPECT_EQ(result[1], 6);
  EXPECT_EQ(result[2], 3);
  EXPECT_EQ(result.dimension, 3);
}

// cf. 12.4 Exercises, Exercise 1d. of Apostol, Calculus, Vol. 1.
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, LeftScalarMultiplicationWorks)
{
  const NVector<3> A {1, 3, 6};
  const NVector<3> D {7.0*A - 2.0*B - 3.0*C};

  EXPECT_EQ(D[0], -7);
  EXPECT_EQ(D[1], 24);
  EXPECT_EQ(D[2], 21);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, AdditionAssignmentWorks)
{
  NVector<3> A {1, 3, 6};

  A += B;

  EXPECT_EQ(A[0], 5);
  EXPECT_EQ(A[1], 0);
  EXPECT_EQ(A[2], 9);
  EXPECT_EQ(A.dimension, 3);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVectorTests, SubtractionAssignmentWorks)
{
  NVector<3> A {1, 3, 6};

  A += B;
  A -= C;

  EXPECT_EQ(A[0], 3);
  EXPECT_EQ(A[1], -1);
  EXPECT_EQ(A[2], 4);
  EXPECT_EQ(A.dimension, 3);
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests