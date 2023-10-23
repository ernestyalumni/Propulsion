#include "IntegrationTests/Visualization/GLFWInterface/SampleCode/fill_RGB.h"

#include <cstddef>

using Visualization::GLFWInterface::GLFWWindow;
using std::size_t;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SampleCode
{

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

__global__ void render_simple_image(
  uchar4* ptr,
  const size_t width,
  const size_t height)
{
  const size_t x {blockIdx.x * blockDim.x + threadIdx.x};
  const size_t y {blockIdx.y * blockDim.y + threadIdx.y};

  if (x >= width || y >= height)
  {
    return;
  }

  const size_t index {y * width + x};

  const uchar4 color {
    make_uchar4(255 * x / width, 255 * y / height, 0, 255)};

  ptr[index] = color;
}

} // namespace SampleCode
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests