#include "gtest/gtest.h"

#include "Utilities/DeviceManagement/GetAndSetDevice.h"
#include "Utilities/HandleUnsuccessfulCUDACall.h"
#include "Testing/Manifolds/Operators/FiniteDifference/2D/DirectionalDerivativesKernel.h"
#include "Testing/Manifolds/Operators/FiniteDifference/2D/Stencil.h"

#include <cuda_runtime.h>

//------------------------------------------------------------------------------
/// Warning: Notice that we did not declare nor implement the template
/// specialization for float2 of the
/// C_nu coefficients cnu_coefficients_first_order. This is because this was
/// already done else where, as of now, in C_Nu_Coefficients_tests.cu. This is
/// because otherwise you get linkage errors in compilation if there's more than
/// one definition.
//------------------------------------------------------------------------------

using Manifolds::Operators::FiniteDifference::set_first_order_coefficients_for_p1;
using Manifolds::Operators::FiniteDifference::set_first_order_coefficients_for_p2;
using Utilities::DeviceManagement::GetAndSetDevice;
using Utilities::HandleUnsuccessfulCUDACall;
using Testing::Manifolds::Operators::FiniteDifference::TwoDimensional::Stencil;
using Testing::Manifolds::Operators::FiniteDifference::TwoDimensional::test_directional_derivatives;

namespace GoogleUnitTests
{
namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

constexpr int device_to_use {1};
//constexpr int device_to_use {0};

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

  host_stencil_1[0][0] = 0.5f;
  host_stencil_1[0][1] = 4.0f;
  stencil_1.copy_host_input_to_device(host_stencil_1);

  const float result_2 {
    test_directional_derivatives<float, float2, 1>(stencil_1.stencil_)};

  EXPECT_FLOAT_EQ(result_2, (4.0f - 0.5f) * 0.5f * 1.0f * 1.0f);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DirectionalDerivativesTests, FirstOrderFloatWithP2CnusCoefficients)
{
  GetAndSetDevice get_and_set_device {};
  get_and_set_device.set_device(device_to_use);

  float hd_i[2] {0.25f, 0.125f};

  set_first_order_coefficients_for_p2<float, float2>(hd_i);

  Stencil<float, 2> stencil_2 {};
  float host_stencil_2[2][2] {};
  host_stencil_2[0][0] = 2.0f;
  host_stencil_2[0][1] = 0.25f;
  host_stencil_2[1][0] = 0.5f;
  host_stencil_2[1][1] = 4.0f;

  stencil_2.copy_host_input_to_device(host_stencil_2);

  const float result_1 {
    test_directional_derivatives<float, float2, 2>(stencil_2.stencil_)};

  EXPECT_FLOAT_EQ(
    result_1,
    (0.25 - 2.f) * (2.f/3.f) * 4.0f +
    (4.0f - 0.5f) * (-1.f/12.f) * 4.0f);

  host_stencil_2[0][0] = 0.5f;
  host_stencil_2[0][1] = 4.0f;
  host_stencil_2[1][0] = 0.25f;
  host_stencil_2[1][1] = 8.0f;

  stencil_2.copy_host_input_to_device(host_stencil_2);

  const float result_2 {
    test_directional_derivatives<float, float2, 2>(stencil_2.stencil_)};

  EXPECT_FLOAT_EQ(
    result_2,
    (4.0f - 0.5f) * (2.f/3.f) * 4.0f +
    (8.0f - 0.25f) * (-1.f/12.f) * 4.0f);
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
} // namespace GoogleUnitTests
