#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_GRAPHICS_INTEROP_CALCULATE_PIXEL_VALUES_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_GRAPHICS_INTEROP_CALCULATE_PIXEL_VALUES_H

#include <cstddef>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace GraphicsInterOp
{

struct Parameters
{
  Parameters(const int dimension, const std::size_t number_of_threads):
    dimension_{dimension},
    number_of_threads_{number_of_threads},
    blocks_per_grid_{
      static_cast<unsigned int>(dimension / number_of_threads),
      static_cast<unsigned int>(dimension / number_of_threads)}
  {}

  int dimension_;
  std::size_t number_of_threads_;
  dim3 blocks_per_grid_;
};

struct CalculatePixelValues
{
  using Parameters = Parameters;

  CalculatePixelValues(const Parameters& parameters):
    parameters_{parameters}
  {}

  void run(uchar4* ptr);

  Parameters parameters_;
};

//------------------------------------------------------------------------------
/// Target buffer is an array of type uchar4.
//------------------------------------------------------------------------------
__global__ void calculate_pixel_value(uchar4* ptr, const int dimension);

} // namespace GraphicsInterOp
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_GRAPHICS_INTEROP_CALCULATE_RIPPLE_VALUES_H