#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_FILL_RGB_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_FILL_RGB_H

#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLFWInterface/GLFWWindow.h"

#include <cstddef>
#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SampleCode
{

__global__ void fill_RGB(unsigned char* rgb);

__global__ void render_simple_image(
  uchar4* ptr,
  const std::size_t width,
  const std::size_t height);

} // namespace SampleCode
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_FILL_RGB_H