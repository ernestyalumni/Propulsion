#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/IsInJuliaSet.h"

#include "DeviceComplexNumber.h"
#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/Parameters.h"

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

__device__ int IsInJuliaSet::is_in_julia_set(
  const int x,
  const int y,
  const float scale)
{
  const float jx {
    scale * static_cast<float>(
      (width_dimension_ / 2 - x) / (width_dimension_ / 2))};

  const float jy {
    scale * static_cast<float>(
      (height_dimension_ / 2 - y) / (height_dimension_ / 2))};

  DeviceComplexNumber c {c_real_, c_imaginary_};
  DeviceComplexNumber a {jx, jy};

  for (int i {0}; i < maximum_iteration_; ++i)
  {
    a = a * a + c;
    if (a.magnitude_squared() > magnitude_threshold_)
    {
      // Return 0 if it's not in the set.
      return 0;
    }
  }
  // Return 1 if point is in set.
  return 1;
}

__global__ void is_in_julia_set(
  uchar4* ptr,
  const float scale,
  const Parameters& parameters)
{
  int x {blockIdx.x};
  int y {blockIdx.y};

  int offset {x + y * gridDim.x};

  IsInJuliaSet is_in_julia_set {parameters};

  // Now calculate the value at that position
  int julia_value {is_in_julia_set(x, y, scale)}

  // Red if is_in_julia_set() returns 1, black if pt. not in set.
  ptr[offset].x = 255 * julia_value;
  ptr[offset].y = 0;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests
