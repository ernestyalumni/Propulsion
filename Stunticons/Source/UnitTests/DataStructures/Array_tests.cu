#include "DataStructures/Array.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using DataStructures::Array;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace DataStructures
{

// We'll use the example code from
// https://forums.developer.nvidia.com/t/using-glfw-library-in-a-cuda-program/248434
// "Using GLFW library in a CUDA program"

__global__ void fill_RGB(unsigned char* rgb)
{
  const size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  const size_t offset {index * 3};
  // Red value
  rgb[offset] = index % 255;
  // Green value
  rgb[offset + 1] = (index * 3) % 255;
  // Blue value
  rgb[offset + 2] = (index * 7) % 255;
}

constexpr size_t example_width {800};
constexpr size_t example_height {600};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArrayTests, Constructible)
{
  Array<unsigned char> array {example_width * example_height * 3};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArrayTests, CopiesFromHostToDevice)
{
  const size_t N {8};

  Array<float> array {N};

  vector<float> source {};
  source.reserve(N);
  std::generate_n(
    std::back_inserter(source),
    N,
    [exponent = 0]() mutable
    {
      return std::pow(2.f, exponent++);
    });

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_EQ(source.at(i), std::pow(2.f, i));    
  }

  ASSERT_EQ(source.size(), N);

  EXPECT_TRUE(array.copy_host_input_to_device(source));

  vector<float> result (N);
  std::fill(result.begin(), result.end(), 1.f);
  array.copy_device_output_to_host(result);

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_EQ(result.at(i), std::pow(2.f, i));
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArrayTests, CUDAKernelFunctionCanMutateArray)
{
  Array<unsigned char> array {example_width * example_height * 3};
  const size_t threads_per_block {256};
  const size_t blocks_per_grid {
    (example_width * example_height + threads_per_block - 1) /
      threads_per_block};

  fill_RGB<<<blocks_per_grid, threads_per_block>>>(array.elements_);

  vector<unsigned char> host_vec_rgb (
    example_width * example_height * 3);

  EXPECT_TRUE(array.copy_device_output_to_host(host_vec_rgb));

  EXPECT_EQ(
    host_vec_rgb.size(),
    example_width * example_height * 3);

  for (size_t i {0}; i < threads_per_block; ++i)
  {
    for (size_t j {0}; j < blocks_per_grid; ++j)
    {
      const size_t index {i + j * threads_per_block};
      const size_t offset {index * 3};

      EXPECT_EQ(host_vec_rgb.at(offset), index % 255);
      EXPECT_EQ(host_vec_rgb.at(offset + 1), (index * 3) % 255);
      EXPECT_EQ(host_vec_rgb.at(offset + 2), (index * 7) % 255);
    }
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArrayTests, CanCopyToHostArray)
{
  Array<unsigned char> array {800 * 600 * 3};
  const size_t threads_per_block {256};
  const size_t blocks_per_grid {
    (800 * 600 + threads_per_block - 1) / threads_per_block};

  fill_RGB<<<blocks_per_grid, threads_per_block>>>(array.elements_);

  unsigned char* host_array = new unsigned char[
    example_width *
      example_height * 3];

  EXPECT_TRUE(array.copy_device_output_to_host(host_array));

  for (size_t i {0}; i < threads_per_block; ++i)
  {
    for (size_t j {0}; j < blocks_per_grid; ++j)
    {
      const size_t index {i + j * threads_per_block};
      const size_t offset {index * 3};

      EXPECT_EQ(host_array[offset], index % 255);
      EXPECT_EQ(host_array[offset + 1], (index * 3) % 255);
      EXPECT_EQ(host_array[offset + 2], (index * 7) % 255);
    }
  }

  delete[] host_array;
}

} // namespace DataStructures
} // namespace GoogleUnitTests