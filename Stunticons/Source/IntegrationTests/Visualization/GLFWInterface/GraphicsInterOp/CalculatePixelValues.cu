#include "CalculatePixelValues.h"

#include <cstddef>

using std::size_t;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace GraphicsInterOp
{

__global__ void calculate_pixel_value(uchar4* ptr, const int dimension)
{
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};
  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  // Now calculate the value at that position.
  const float fx {static_cast<float>(static_cast<int>(x) - dimension / 2)};
  const float fy {static_cast<float>(static_cast<int>(y) - dimension / 2)};

  const unsigned char green {
    static_cast<unsigned char>(
      128.0f + 127.0f * sinf(fabsf(fx * 100) - fabsf(fy * 100)))};

  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

void CalculatePixelValues::run(uchar4* pixels)
{
  const dim3 threads_per_block {
    static_cast<unsigned int>(parameters_.number_of_threads_),
    static_cast<unsigned int>(parameters_.number_of_threads_)};

  calculate_pixel_value<<<parameters_.blocks_per_grid_, threads_per_block>>>(
    pixels,
    parameters_.dimension_);
}

} // namespace GraphicsInterOp
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests
