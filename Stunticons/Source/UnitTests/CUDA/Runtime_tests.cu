//------------------------------------------------------------------------------
/// \brief Test and demonstrate features in CUDA runtime (cuda_runtime.h) in
/// order to clarify its operation and what they do, especially when
/// documentation and examples are lacking.
//------------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <cuda_runtime.h>

namespace GoogleUnitTests
{
namespace CUDA
{

//------------------------------------------------------------------------------
/// From cuda-samples, 0 Introduction, simpleCUDA2GLU.cu.
//------------------------------------------------------------------------------
uchar4 make_uchar4_example(const int x, const int y)
{
  return make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);	
}

//------------------------------------------------------------------------------
/// TODO: I couldn't finding documentation online for make_uchar4. At most it
/// was said to be a macro in cuda_runtime.h.
//------------------------------------------------------------------------------
TEST(RuntimeTests, MakeUchar4Makes)
{
  const uchar4 c41 {make_uchar4(1, 2, 3, 4)};

  EXPECT_EQ(c41.x, 1);
  EXPECT_EQ(c41.y, 2);
  EXPECT_EQ(c41.z, 3);
  EXPECT_EQ(c41.w, 4);
}

} // namespace CUDA
} // namespace GoogleUnitTests