#include "DataStructures/Array.h"
#include "Utilities/HandleUnsuccessfulCudaCall.h"
#include "Visualization/float_to_color.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using DataStructures::Array;
using Utilities::HandleUnsuccessfulCUDACall;
using Visualization::ColorConversion::float_is_saturation;
using Visualization::ColorConversion::float_to_color;
using Visualization::ColorConversion::float_to_color_with_linear_saturation;
using Visualization::ColorConversion::float_to_color_with_set_saturation;
using Visualization::ColorConversion::linear_float_to_saturation;
using Visualization::ColorConversion::to_RGB;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace Visualization
{

__global__ void identity_in_1d(float* target, const float* source)
{
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};

  target[x] = float_is_saturation(source[x]);
}

__global__ void identity_in_2d(float* target, const float* source)
{
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  target[offset] = float_is_saturation(source[offset]);
}

void run_identity_in_1d(
  float* target,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  identity_in_1d<<<blocks_per_grid, threads_per_block>>>(target, source);
} 

void run_identity_in_2d(
  float* target,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  identity_in_2d<<<blocks_per_grid, threads_per_block>>>(target, source);
} 

__global__ void linear_float_to_saturation_in_1d(
  float* target,
  const float* source)
{
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};

  target[x] = linear_float_to_saturation(source[x]);
}

__global__ void linear_float_to_saturation_in_2d(
  float* target,
  const float* source)
{
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  target[offset] = linear_float_to_saturation(source[offset]);
}

void run_linear_float_to_saturation_in_1d(
  float* target,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  linear_float_to_saturation_in_1d<<<blocks_per_grid, threads_per_block>>>(
    target,
    source);
} 

void run_linear_float_to_saturation_in_2d(
  float* target,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  linear_float_to_saturation_in_2d<<<blocks_per_grid, threads_per_block>>>(
    target,
    source);
} 

void run_float_to_color_with_set_saturation(
  unsigned char* optr,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  float_to_color_with_set_saturation<<<
    blocks_per_grid,
    threads_per_block>>>(
      optr,
      source);
}

void run_float_to_color_with_set_saturation(
  uchar4* optr,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  float_to_color_with_set_saturation<<<
    blocks_per_grid,
    threads_per_block>>>(
      optr,
      source);
}

void run_float_to_color_with_linear_saturation(
  unsigned char* optr,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  float_to_color_with_linear_saturation<<<
    blocks_per_grid,
    threads_per_block>>>(
      optr,
      source);
}

void run_float_to_color_with_linear_saturation(
  uchar4* optr,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  float_to_color_with_linear_saturation<<<
    blocks_per_grid,
    threads_per_block>>>(
      optr,
      source);
}

// TODO: This files compilation because at compilation it needs the definition
// of to_RGB inline'ed in the header.
/*
bool test_linear_float_to_color(
  unsigned char* optr,
  const float* source,
  const dim3 blocks_per_grid,
  const dim3 threads_per_block)
{
  float_to_color<<<blocks_per_grid, threads_per_block>>>(
    optr,
    source,
    linear_float_to_saturation);

  HandleUnsuccessfulCUDACall handle_synchronization {
    "Failed to synchronize device(s)"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_synchronization,
    cudaDeviceSynchronize());

  return handle_synchronization.is_cuda_success();
}
*/

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FloatToColorTests, FloatIsSaturationIsIdentityIn1D)
{
  const size_t N {32};
  Array<float> source_array {N};
  vector<float> source (N);
  std::generate_n(
    source.begin(),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::sort(source.begin(), source.end());
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<float> target (N);
  std::fill(target.begin(), target.end(), 42.f);
  Array<float> target_array {N};
  ASSERT_TRUE(target_array.copy_host_input_to_device(target));

  run_identity_in_1d(
    target_array.elements_,
    source_array.elements_,
    dim3{1},
    dim3{N});

  vector<float> result (N);

  ASSERT_TRUE(target_array.copy_device_output_to_host(result));

  int exponent {0};
  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(result.at(i), std::pow(2.f, exponent));

    exponent++;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FloatToColorTests, FloatIsSaturationIsIdentityIn2D)
{
  const size_t N {8 * 16};
  Array<float> source_array {N};
  vector<float> source (N);
  std::generate_n(
    source.begin(),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::sort(source.begin(), source.end());
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<float> target (N);
  std::fill(target.begin(), target.end(), 42.f);
  Array<float> target_array {N};
  ASSERT_TRUE(target_array.copy_host_input_to_device(target));

  run_identity_in_2d(
    target_array.elements_,
    source_array.elements_,
    dim3{1, 1},
    dim3{8, 16});

  vector<float> result (N);

  ASSERT_TRUE(target_array.copy_device_output_to_host(result));

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(result.at(i), std::pow(2.f, i));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FloatToColorTests, LinearFloatToSaturationComputesIn1D)
{
  const size_t N {128};
  Array<float> source_array {N};
  vector<float> source (N);
  std::generate_n(
    source.begin(),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::sort(source.begin(), source.end());
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<float> target (N);
  std::fill(target.begin(), target.end(), 42.f);
  Array<float> target_array {N};
  ASSERT_TRUE(target_array.copy_host_input_to_device(target));

  run_linear_float_to_saturation_in_1d(
    target_array.elements_,
    source_array.elements_,
    dim3{N / 4},
    dim3{4});

  vector<float> result (N);

  ASSERT_TRUE(target_array.copy_device_output_to_host(result));

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(
      result.at(i),
      1.0f - std::abs(0.5f - std::pow(2.f, i)) * 2.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FloatToColorTests, LinearFloatToSaturationComputesIn2D)
{
  // 512
  const size_t N {16 * 32};
  Array<float> source_array {N};
  vector<float> source (N);
  std::generate_n(
    source.begin(),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::sort(source.begin(), source.end());
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<float> target (N);
  std::fill(target.begin(), target.end(), 42.f);
  Array<float> target_array {N};
  ASSERT_TRUE(target_array.copy_host_input_to_device(target));

  run_linear_float_to_saturation_in_2d(
    target_array.elements_,
    source_array.elements_,
    dim3{16 / 4, 32 / 2},
    dim3{4, 2});

  vector<float> result (N);

  ASSERT_TRUE(target_array.copy_device_output_to_host(result));

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(
      result.at(i),
      1.0f - std::abs(0.5f - std::pow(2.f, i)) * 2.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  FloatToColorTests,
  FloatToColorWithSetSaturationWorksOnPositiveMultiplesOf2)
{
  const size_t N {32 * 16};
  Array<float> source_array {N};
  vector<float> source (N);
  std::generate_n(
    source.begin(),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<unsigned char> target (4 * N);
  std::fill(target.begin(), target.end(), static_cast<unsigned char>(242));
  Array<unsigned char> target_array {N * 4};
  ASSERT_TRUE(target_array.copy_host_input_to_device(target));

  run_float_to_color_with_set_saturation(
    target_array.elements_,
    source_array.elements_,
    dim3{32 / 4, 16 / 8},
    dim3{4, 8});

  vector<unsigned char> result (4 * N);
  std::fill(result.begin(), result.end(), static_cast<unsigned char>(69));
  ASSERT_TRUE(target_array.copy_device_output_to_host(result));

  EXPECT_EQ(result.at(0), 255);
  EXPECT_EQ(result.at(1), 255);
  EXPECT_EQ(result.at(2), 255);
  EXPECT_EQ(result.at(3), 255);
  EXPECT_EQ(result.at(4), 252);
  EXPECT_EQ(result.at(5), '\0');
  EXPECT_EQ(result.at(6), '\0');
  EXPECT_EQ(result.at(7), 255);
  EXPECT_EQ(result.at(4 * N - 4), '\0');
  EXPECT_EQ(result.at(4 * N - 3), '\0');
  EXPECT_EQ(result.at(4 * N - 2), '\0');
  EXPECT_EQ(result.at(4 * N - 1), 255);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FloatToColorTests, FloatToColorWithSetSaturationWorksWithUchar4)
{
  const size_t N {64 * 16};
  Array<float> source_array {2 * N};
  vector<float> source (2 * N);

  std::generate_n(
    source.begin() + N,
    N,
    [exponent = 1]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::transform(
    source.begin() + N,
    source.end(),
    source.begin(),
    [](float value)
    {
      return -value;
    });

  std::sort(source.begin(), source.end());
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<uchar4> target (2 * N, {42, 69, 43, 70});
  Array<uchar4> optr {2 * N};
  ASSERT_TRUE(optr.copy_host_input_to_device(target));

  run_float_to_color_with_set_saturation(
    optr.elements_,
    source_array.elements_,
    dim3{64 / 4, 32 / 8},
    dim3{4, 8});

  vector<uchar4> result (2 * N);
  ASSERT_TRUE(optr.copy_device_output_to_host(result));

  EXPECT_EQ(result.at(0).x, 255);
  EXPECT_EQ(result.at(0).y, '\0');
  EXPECT_EQ(result.at(0).z, '\0');
  EXPECT_EQ(result.at(0).w, 255);
  EXPECT_EQ(result.at(1).x, 255);
  EXPECT_EQ(result.at(1).y, '\0');
  EXPECT_EQ(result.at(1).z, '\0');
  EXPECT_EQ(result.at(1).w, 255);
  EXPECT_EQ(result.at(N + 1).x, 240);
  EXPECT_EQ(result.at(N + 1).y, '\0');
  EXPECT_EQ(result.at(N + 1).z, '\0');
  EXPECT_EQ(result.at(N + 1).w, 255);
  EXPECT_EQ(result.at(N + 2).x, 192);
  EXPECT_EQ(result.at(N + 2).y, '\0');
  EXPECT_EQ(result.at(N + 2).z, '\0');
  EXPECT_EQ(result.at(N + 2).w, 255);
  EXPECT_EQ(result.at(2 * N - 1).x, '\0');
  EXPECT_EQ(result.at(2 * N - 1).y, '\0');
  EXPECT_EQ(result.at(2 * N - 1).z, '\0');
  EXPECT_EQ(result.at(2 * N - 1).w, 255);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FloatToColorTests, FloatToColorWithLinearSaturationWorksWithUnsignedChars)
{
  const size_t N {128 * 32};
  Array<float> source_array {2 * N};
  vector<float> source (2 * N);

  std::generate_n(
    source.begin() + N,
    N,
    [exponent = 1]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::transform(
    source.begin() + N,
    source.end(),
    source.begin(),
    [](float value)
    {
      return -value;
    });

  std::sort(source.begin(), source.end());
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<unsigned char> target (4 * 2 * N);
  std::fill(target.begin(), target.end(), static_cast<unsigned char>(169));

  Array<unsigned char> target_array {4 * 2 * N};
  ASSERT_TRUE(target_array.copy_host_input_to_device(target));

  run_float_to_color_with_linear_saturation(
    target_array.elements_,
    source_array.elements_,
    dim3{128 / 16, 64 / 8},
    dim3{16, 8});

  vector<unsigned char> result (4 * 2 * N);
  std::fill(result.begin(), result.end(), static_cast<unsigned char>(42));

  ASSERT_TRUE(target_array.copy_device_output_to_host(result));

  EXPECT_EQ(result.at(0), 255);
  EXPECT_EQ(result.at(1), '\0');
  EXPECT_EQ(result.at(2), '\0');
  EXPECT_EQ(result.at(3), 255);
  EXPECT_EQ(result.at(4), 255);
  EXPECT_EQ(result.at(N + 1), '\0');
  EXPECT_EQ(result.at(N + 2), '\0');
  EXPECT_EQ(result.at(N + 3), 255);
  EXPECT_EQ(result.at(N + 4), 255);
  EXPECT_EQ(result.at(2 * N + 1), '\0');
  EXPECT_EQ(result.at(2 * N + 2), '\0');
  EXPECT_EQ(result.at(2 * N + 3), 255);
  EXPECT_EQ(result.at(2 * N + 4), 255);
  EXPECT_EQ(result.at(4 * 2 * N - 4), '\0');
  EXPECT_EQ(result.at(4 * 2 * N - 3), '\0');
  EXPECT_EQ(result.at(4 * 2 * N - 2), '\0');
  EXPECT_EQ(result.at(4 * 2 * N - 1), 255);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FloatToColorTests, FloatToColorWithLinearSaturationWorksWithUchar4)
{
  const size_t N {256 * 64};
  Array<float> source_array {N};
  vector<float> source (N);
  std::generate_n(
    source.begin(),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });
  ASSERT_TRUE(source_array.copy_host_input_to_device(source));

  vector<uchar4> target (N);
  std::fill(target.begin(), target.end(), uchar4{68, 41, 69, 42});
  Array<uchar4> target_array {N};

  ASSERT_TRUE(target_array.copy_host_input_to_device(target));

  run_float_to_color_with_linear_saturation(
    target_array.elements_,
    source_array.elements_,
    dim3{256 / 16, 64 / 8},
    dim3{16, 8});

  vector<uchar4> result (N);
  ASSERT_TRUE(target_array.copy_device_output_to_host(result));

  EXPECT_EQ(result.at(0).x, 255);
  EXPECT_EQ(result.at(0).y, 255);
  EXPECT_EQ(result.at(0).z, 255);
  EXPECT_EQ(result.at(0).w, 255);
  EXPECT_EQ(result.at(1).x, '\0');
  EXPECT_EQ(result.at(1).y, 252);
  EXPECT_EQ(result.at(1).z, 252);
  EXPECT_EQ(result.at(1).w, 255);
  EXPECT_EQ(result.at(N / 2 - 2).x, '\0');
  EXPECT_EQ(result.at(N / 2 - 2).y, '\0');
  EXPECT_EQ(result.at(N / 2 - 2).z, '\0');
  EXPECT_EQ(result.at(N / 2 - 2).w, 255);
  EXPECT_EQ(result.at(N - 2).x, '\0');
  EXPECT_EQ(result.at(N - 2).y, '\0');
  EXPECT_EQ(result.at(N - 2).z, '\0');
  EXPECT_EQ(result.at(N - 2).w, 255);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
/*
TEST(
  FloatToColorTests,
  FloatToColorEquatingLightnessToSaturationWorksOnMultiplesOf2)
{
  const size_t N {16 * 8};

  Array<float> array {N};

  vector<float> source (N);
  /*
  std::generate_n(
    source.begin() + N,
    N,
    [exponent = 1]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::transform(
    source.begin() + N,
    source.end(),
    source.begin(),
    [](float value)
    {
      return -value;
    });
  */

  /*
  std::generate_n(
    source.begin(),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  std::sort(source.begin(), source.end());

  ASSERT_EQ(source.size(), N);
  ASSERT_TRUE(array.copy_host_input_to_device(source));

  Array<unsigned char> optr {N * 4};

  EXPECT_TRUE(test_linear_float_to_color(
    optr.elements_,
    array.elements_,
    dim3{16 / 8, 8 / 4},
    dim3{8, 4}));
}
*/

} // namespace Visualization
} // namespace GoogleUnitTests