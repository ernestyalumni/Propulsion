#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_RIPPLE_CALCULATE_RIPPLE_VALUE_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_RIPPLE_CALCULATE_RIPPLE_VALUE_H

#include <cstddef>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace Ripple
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

//------------------------------------------------------------------------------
/// Target buffer is an array of type uchar4.
//------------------------------------------------------------------------------
__global__ void calculate_ripple_value(
  uchar4* ptr,
  const int ticks,
  const Parameters parameters);

void calculate_ripple_values(
  uchar4* pixels,
  const int ticks,
  const Parameters parameters);

//------------------------------------------------------------------------------
/// Original code
//------------------------------------------------------------------------------
void generate_frame(uchar4 *pixels, void*, int ticks);

} // namespace Ripple
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_RIPPLE_CALCULATE_RIPPLE_VALUE_H