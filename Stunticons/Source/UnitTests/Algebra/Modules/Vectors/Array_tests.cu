#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/HostArrays.h"
#include "gtest/gtest.h"

#include <cstddef>

using Algebra::Modules::Vectors::Array;
using Algebra::Modules::Vectors::HostArray;

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
TEST(ArrayTests, Constructible)
{
  static constexpr std::size_t N {1048576};

  Array x {N};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArrayTests, CopiesToDevice)
{
  static constexpr std::size_t N {1048576};

  HostArray x {N};

  for (size_t i {0}; i < N; ++i)
  {
    x.values_[i] = static_cast<float>(i) + 0.42f;
  }

  Array d_x {N};
  EXPECT_TRUE(d_x.copy_host_input_to_device(x));

  HostArray x_out {N};

  for (size_t i {0}; i < N; ++i)
  {
    x_out.values_[i] = 69.0f;
  }

  EXPECT_TRUE(d_x.copy_device_output_to_host(x_out));

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(x_out.values_[i], static_cast<float>(i) + 0.42f);
  }
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests