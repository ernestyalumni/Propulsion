#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/IsInJuliaSet.h"

#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/DeviceComplexNumber.h"
#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/Parameters.h"

#include <cstddef>

using std::size_t;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{
namespace JuliaSet
{

#define DIM 1500
#define MAG_THR 1000 // magnitude threshold that determines if pt. is in Julia set
#define TESTITERS 300 // originally 200, tests further what points go to infinity; higher no. makes it "lacy"

// Constants that change formula for f, f(z) = z*z + c
#define CREAL -0.8168 // originally -0.8
#define CIMAG 0.1583 // originally 0.154

__device__ int julia(
  const int x,
  const int y,
  const float scale,
  const int width,
  const int height)
{
  const float jx {
    scale * static_cast<float>(
      (width / 2 - x)) / static_cast<float>(width / 2)};

  const float jy {
    scale * static_cast<float>(
      (height / 2 - y)) / static_cast<float>(height / 2)};

  /*
  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
  float jy = scale * (float)(DIM/2 - y)/(DIM/2);
  */

  DeviceComplexNumber c {CREAL, CIMAG};
  DeviceComplexNumber a {jx, jy};

  for (int i {0}; i<TESTITERS; i++) {
    a = a*a + c;
    if (a.magnitude_squared() > MAG_THR)
      return 0; // return 0 if it is not in set
  }
  return 1; // return 1 if point is in set
}

__device__ int julia(
  const int x,
  const int y,
  const float scale,
  Parameters parameters)
{
  const float jx {
    scale * static_cast<float>(
      (parameters.width_dimension_ / 2 - x)) /
        static_cast<float>(parameters.width_dimension_ / 2)};

  const float jy {
    scale * static_cast<float>(
      (parameters.height_dimension_ / 2 - y)) /
        static_cast<float>(parameters.height_dimension_ / 2)};

  DeviceComplexNumber c {CREAL, CIMAG};
  DeviceComplexNumber a {jx, jy};

  for (int i {0}; i < parameters.maximum_iterations_; i++)
  {
    a = a*a + c;

    if (a.magnitude_squared() > parameters.magnitude_threshold_)
    {
      // return 0 if it is not in set
      return 0; 
    }
  }
  
  // return 1 if point is in set
  return 1;
}

__device__ int IsInJuliaSet::is_in_julia_set(
  const int x,
  const int y,
  const float scale)
{
  const float jx {
    scale * static_cast<float>(
      (width_dimension_ / 2 - x)) /
        static_cast<float>(width_dimension_ / 2)};

  const float jy {
    scale * static_cast<float>(
      (height_dimension_ / 2 - y)) /
        static_cast<float>(height_dimension_ / 2)};

  DeviceComplexNumber c {c_real_, c_imaginary_};
  DeviceComplexNumber a {jx, jy};

  for (int i {0}; i < maximum_iterations_; ++i)
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
  const Parameters parameters)
{
  size_t x {blockIdx.x};
  size_t y {blockIdx.y};

  size_t offset {x + y * gridDim.x};

  IsInJuliaSet is_in_julia_set {parameters};

  // Now calculate the value at that position
  int julia_value {is_in_julia_set.is_in_julia_set(x, y, scale)};

  // Previously used sanity check.
  /*
  int julia_value {
    julia(
      x,
      y,
      scale,
      parameters)};
  */
      //is_in_julia_set.get_width_dimension(),
      //is_in_julia_set.get_height_dimension())};

  // Red if is_in_julia_set() returns 1, black if pt. not in set.
  ptr[offset].x = 255 * julia_value;
  ptr[offset].y = 0;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

} // namespace JuliaSet
} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests
