#include "gtest/gtest.h"

#include "Manifolds/Operators/FiniteDifference/DirectionalDerivatives.h"
#include "Utilities/DeviceManagement/GetAndSetDevice.h"
#include "Utilities/HandleUnsuccessfulCUDACall.h"
#include "Utilities/Testing/Manifolds/Operators/FiniteDifference/DirectionalDerivativesKernel.h"
#include "Utilities/Testing/Manifolds/Operators/FiniteDifference/Stencil.h"

#include <cuda_runtime.h>

using Manifolds::Operators::FiniteDifference::set_first_order_coefficients_for_p1;
using Utilities::DeviceManagement::GetAndSetDevice;
using Utilities::HandleUnsuccessfulCUDACall;
using Utilities::Testing::Manifolds::Operators::FiniteDifference::Stencil;
using Utilities::Testing::Manifolds::Operators::FiniteDifference::test_directional_derivatives;

namespace GoogleUnitTests
{
namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

//constexpr int device_to_use {1};
constexpr int device_to_use {0};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DirectionalDerivativesTests, FirstOrderFloat)
{
  GetAndSetDevice get_and_set_device {};
  get_and_set_device.set_device(device_to_use);

  float hd_i[2] {1.0f, 0.5f};

  set_first_order_coefficients_for_p1<float, float2>(hd_i);

  Stencil<float, 1> stencil_1 {};
  float host_stencil_1[1][2] {};
  host_stencil_1[0][0] = 2.0f;
  host_stencil_1[0][1] = 0.25f;
  stencil_1.copy_host_input_to_device(host_stencil_1);

  const float result_1 {
    test_directional_derivatives<float, float2, 1>(stencil_1.stencil_)};

  EXPECT_FLOAT_EQ(result_1, (0.25 - 2.f) * 0.5f * 1.0f * 1.0f);
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
} // namespace GoogleUnitTests
