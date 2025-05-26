#include "gtest/gtest.h"

#include "Manifolds/Operators/FiniteDifference/C_Nu_Coefficients.h"
#include "Utilities/DeviceManagement/GetAndSetDevice.h"
#include "Utilities/HandleUnsuccessfulCUDACall.h"

#include <cuda_runtime.h>

using Manifolds::Operators::FiniteDifference::cnu_coefficients_first_order;
using Manifolds::Operators::FiniteDifference::cnu_coefficients_second_order;
using Manifolds::Operators::FiniteDifference::set_first_order_coefficients_for_p1;
using Manifolds::Operators::FiniteDifference::set_first_order_coefficients_for_p2;
using Manifolds::Operators::FiniteDifference::set_second_order_coefficients_for_p3;
using Manifolds::Operators::FiniteDifference::set_second_order_coefficients_for_p4;
using Utilities::DeviceManagement::GetAndSetDevice;
using Utilities::HandleUnsuccessfulCUDACall;

template<>
__constant__ float2 cnu_coefficients_first_order<float2>[4];

template<>
__constant__ float2 cnu_coefficients_second_order<float2>[4];

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

Utilities::HandleUnsuccessfulCUDACall handle_memcpy_from_symbol {
"Failed to copy c_nus from symbol on device"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(C_Nu_CoefficientsTests, OrderP1Constructible)
{
  GetAndSetDevice get_and_set_device {};
  get_and_set_device.set_device(device_to_use);

  float hd_i[2] {1.0f, 0.5f};

  set_first_order_coefficients_for_p1<float, float2>(hd_i);

  float2 host_cnus[4] {};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_from_symbol,
    cudaMemcpyFromSymbol(
      host_cnus,
      cnu_coefficients_first_order<float2>,
      sizeof(float2) * 4,
      0,
      cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(host_cnus[0].x, 0.5f * 1.0f);
  EXPECT_FLOAT_EQ(host_cnus[0].y, 0.5f * 2.0f);

  for (int i {1}; i < 4; ++i)
  {
    EXPECT_FLOAT_EQ(host_cnus[i].x, 0.0f);
    EXPECT_FLOAT_EQ(host_cnus[i].y, 0.0f);
  }

  hd_i[0] = 0.25f;
  hd_i[1] = 0.125f;

  set_first_order_coefficients_for_p1<float, float2>(hd_i);

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_from_symbol,
    cudaMemcpyFromSymbol(
      host_cnus,
      cnu_coefficients_first_order<float2>,
      sizeof(float2) * 4,
      0,
      cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(host_cnus[0].x, 0.5f * 4.0f);
  EXPECT_FLOAT_EQ(host_cnus[0].y, 0.5f * 8.0f);

  for (int i {1}; i < 4; ++i)
  {
    EXPECT_FLOAT_EQ(host_cnus[i].x, 0.0f);
    EXPECT_FLOAT_EQ(host_cnus[i].y, 0.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(C_Nu_CoefficientsTests, OrderP2Constructible)
{
  GetAndSetDevice get_and_set_device {};
  get_and_set_device.set_device(device_to_use);

  float hd_i[2] {1.0f, 0.5f};

  set_first_order_coefficients_for_p2<float, float2>(hd_i);

  float2 host_cnus[4] {};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_from_symbol,
    cudaMemcpyFromSymbol(
      host_cnus,
      cnu_coefficients_first_order<float2>,
      sizeof(float2) * 4,
      0,
      cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(host_cnus[0].x, 2.f / 3.f * 1.0f);
  EXPECT_FLOAT_EQ(host_cnus[0].y, 2.f / 3.f * 2.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].x, -1. / 12.f * 1.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].y, -1. / 12.f * 2.0f);

  for (int i {2}; i < 4; ++i)
  {
    EXPECT_FLOAT_EQ(host_cnus[i].x, 0.0f);
    EXPECT_FLOAT_EQ(host_cnus[i].y, 0.0f);
  }

  hd_i[0] = 0.25f;
  hd_i[1] = 0.125f;

  set_first_order_coefficients_for_p2<float, float2>(hd_i);

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_from_symbol,
    cudaMemcpyFromSymbol(
      host_cnus,
      cnu_coefficients_first_order<float2>,
      sizeof(float2) * 4,
      0,
      cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(host_cnus[0].x, 2.f / 3.f * 4.0f);
  EXPECT_FLOAT_EQ(host_cnus[0].y, 2.f / 3.f * 8.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].x, -1. / 12.f * 4.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].y, -1. / 12.f * 8.0f);

  for (int i {2}; i < 4; ++i)
  {
    EXPECT_FLOAT_EQ(host_cnus[i].x, 0.0f);
    EXPECT_FLOAT_EQ(host_cnus[i].y, 0.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  C_Nu_CoefficientsTests,
  SecondOrderP3ConstructibleAndSecondOrderP4ConstructibleAfterwards)
{
  GetAndSetDevice get_and_set_device {};
  get_and_set_device.set_device(device_to_use);

  float hd_i[2] {1.0f, 0.5f};

  set_second_order_coefficients_for_p3<float, float2>(hd_i);

  float2 host_cnus[4] {};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_from_symbol,
    cudaMemcpyFromSymbol(
      host_cnus,
      cnu_coefficients_second_order<float2>,
      sizeof(float2) * 4,
      0,
      cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(host_cnus[0].x, 3.f / 2.f * 1.0f);
  EXPECT_FLOAT_EQ(host_cnus[0].y, 3.f / 2.f * 4.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].x, -3. / 20.f * 1.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].y, -3. / 20.f * 4.0f);
  EXPECT_FLOAT_EQ(host_cnus[2].x, 1. / 90.f * 1.0f);
  EXPECT_FLOAT_EQ(host_cnus[2].y, 1. / 90.f * 4.0f);

  for (int i {3}; i < 4; ++i)
  {
    EXPECT_FLOAT_EQ(host_cnus[i].x, 0.0f);
    EXPECT_FLOAT_EQ(host_cnus[i].y, 0.0f);
  }

  hd_i[0] = 0.25f;
  hd_i[1] = 0.125f;

  set_second_order_coefficients_for_p4<float, float2>(hd_i);

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_from_symbol,
    cudaMemcpyFromSymbol(
      host_cnus,
      cnu_coefficients_second_order<float2>,
      sizeof(float2) * 4,
      0,
      cudaMemcpyDeviceToHost));

  EXPECT_FLOAT_EQ(host_cnus[0].x, 8.f / 5.f * 16.0f);
  EXPECT_FLOAT_EQ(host_cnus[0].y, 8.f / 5.f * 64.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].x, -1. / 5.f * 16.0f);
  EXPECT_FLOAT_EQ(host_cnus[1].y, -1. / 5.f * 64.0f);
  EXPECT_FLOAT_EQ(host_cnus[2].x, 8. / 315.f * 16.0f);
  EXPECT_FLOAT_EQ(host_cnus[2].y, 8. / 315.f * 64.0f);
  EXPECT_FLOAT_EQ(host_cnus[3].x, -1. / 560.f * 16.0f);
  EXPECT_FLOAT_EQ(host_cnus[3].y, -1. / 560.f * 64.0f);
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
} // namespace GoogleUnitTests

