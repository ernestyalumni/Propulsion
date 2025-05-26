#include "gtest/gtest.h"

#include "Utilities/DeviceManagement/GetAndSetDevice.h"
#include "Utilities/Testing/Manifolds/Operators/FiniteDifference/Stencil.h"

using Utilities::DeviceManagement::GetAndSetDevice;
using Utilities::Testing::Manifolds::Operators::FiniteDifference::Stencil;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Testing
{

//constexpr int device_to_use {1};
constexpr int device_to_use {0};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StencilTests, Constructs)
{
  GetAndSetDevice get_and_set_device {};
  get_and_set_device.set_device(device_to_use);

  Stencil<float, 1> stencil_1 {};
  EXPECT_FALSE(stencil_1.is_cuda_freed_);

  Stencil<float, 2> stencil_2 {};
  EXPECT_FALSE(stencil_2.is_cuda_freed_);

  Stencil<float, 3> stencil_3 {};
  EXPECT_FALSE(stencil_3.is_cuda_freed_);

  Stencil<float, 4> stencil_4 {};
  EXPECT_FALSE(stencil_4.is_cuda_freed_);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StencilTests, CopiesToDevice)
{
  GetAndSetDevice get_and_set_device {};
  get_and_set_device.set_device(device_to_use);

  float host_stencil_1[1][2] {};
  host_stencil_1[0][0] = 1.0f;
  host_stencil_1[0][1] = 0.5f;

  float host_stencil_2[2][2] {};
  host_stencil_2[0][0] = 1.0f;
  host_stencil_2[0][1] = 0.5f;
  host_stencil_2[1][0] = 2.0f;
  host_stencil_2[1][1] = 0.25f;
  
  Stencil<float, 1> stencil_1 {};
  EXPECT_TRUE(stencil_1.copy_host_input_to_device(host_stencil_1));

  float device_stencil_1[1][2] {};
  EXPECT_TRUE(stencil_1.copy_device_output_to_host(device_stencil_1));
  EXPECT_FLOAT_EQ(device_stencil_1[0][0], 1.0f);
  EXPECT_FLOAT_EQ(device_stencil_1[0][1], 0.5f);

  Stencil<float, 2> stencil_2 {};
  EXPECT_TRUE(stencil_2.copy_host_input_to_device(host_stencil_2));

  float device_stencil_2[2][2] {};
  EXPECT_TRUE(stencil_2.copy_device_output_to_host(device_stencil_2));
  EXPECT_FLOAT_EQ(device_stencil_2[0][0], 1.0f);
  EXPECT_FLOAT_EQ(device_stencil_2[0][1], 0.5f);
  EXPECT_FLOAT_EQ(device_stencil_2[1][0], 2.0f);
  EXPECT_FLOAT_EQ(device_stencil_2[1][1], 0.25f);
}

} // namespace Testing
} // namespace Utilities
} // namespace GoogleUnitTests