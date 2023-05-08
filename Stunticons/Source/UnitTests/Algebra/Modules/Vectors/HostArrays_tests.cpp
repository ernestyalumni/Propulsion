#include "Algebra/Modules/Vectors/HostArrays.h"
#include "gtest/gtest.h"

#include <cstddef>

using Algebra::Modules::Vectors::HostArray;
using std::size_t;

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
TEST(HostArrayTests, Constructible)
{
  static constexpr std::size_t N {1048576};

  HostArray x {N};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostArrayTests, CanBeMutated)
{
  static constexpr std::size_t N {1048576};

  HostArray x {N};
  HostArray rhs {N};

  for (size_t i {0}; i < N; ++i)
  {
    rhs.values_[i] = 1.0;
    x.values_[i] = 0.0;
  }

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(rhs.values_[i], 1.0);
    EXPECT_FLOAT_EQ(x.values_[i], 0.0);
  }
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests