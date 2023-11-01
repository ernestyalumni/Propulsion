#include "calculate_temperature.h"

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SimpleHeat
{

__constant__ DeviceParameters device_parameters;

__global__ void copy_constant_array(float* input, const float* constant_input)
{
  // Map from threadIdx/blockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  // Notice that copy is performed only if cell in constant grid is nonzero
  // We do this to preserve any values computed in previous time step within
  // cells that don't contain heaters.
  if (constant_input[offset] != 0)
  {
    input[offset] = constant_input[offset];
  }
}

__global__ void calculate_temperature(float* output, const float* input)
{
  // Map from threadIdx/blockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  const unsigned int left {x == 0 ? offset : offset - 1};
  const unsigned int right {x == (device_parameters.width_) - 1 ? offset :
    offset + 1};

  // Boundary conditions.
  // Remember that to "move" or "go up" or "go down" 1 unit, one must "cross"
  // over dimensions_ number of elements to the next row up or down.
  const unsigned int top {y == 0 ? offset : offset - device_parameters.width_};
  const unsigned int bottom {y == (device_parameters.height_ -1) ?
    offset : offset + device_parameters.width_};  

  // Update step.
  output[offset] = input[offset] + device_parameters.speed_ * (input[top] +
    input[bottom] + 
    input[left] +
    input[right] -
    input[offset] * 4);
}

} // namespace SimpleHeat
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests
