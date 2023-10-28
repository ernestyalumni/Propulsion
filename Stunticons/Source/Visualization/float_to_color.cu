#include "float_to_color.h"

#include <cuda_runtime.h> // fabsf

namespace Visualization
{
namespace ColorConversion
{

__device__ unsigned char to_RGB(const float n1, const float n2, const int h)
{
  int hue {h};

  if (hue > 360)
  {
    hue -= 360;
  }
  else if (hue < 0)
  {
    hue += 360;
  }

  if (hue < 60)
  {
    return static_cast<unsigned char>(255 * (n1 + (n2 - n1) * hue / 60));
  }

  if (hue < 180)
  {
    return static_cast<unsigned char>(255 * n2);
  }

  if (hue < 240)
  {
    return static_cast<unsigned char>(
      255 * (n1 + (n2 - n1) * (240 - hue) / 60));
  }

  return static_cast<unsigned char>(255 * n1);
}

__global__ void float_to_color_with_set_saturation(
  unsigned char* optr,
  const float* source)
{
  // Map from threadIdx, BlockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  const float l {source[offset]};
  const float s {l};

  const int h {(180 + static_cast<int>(360.f * source[offset])) % 360};

  const float m2 {(l <= 0.5f) ? l * (l + s) : l + s - l * s};

  const float m1 {2 * l - m2};

  optr[offset * 4 + 0] = to_RGB(m1, m2, h + 120);
  optr[offset * 4 + 1] = to_RGB(m1, m2, h);
  optr[offset * 4 + 2] = to_RGB(m1, m2, h - 120);
  optr[offset * 4 + 3] = 255;
}

__global__ void float_to_color_with_set_saturation(
  uchar4* optr,
  const float* source)
{
  // Map from threadIdx, BlockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  const float l {source[offset]};
  const float s {l};

  const int h {(180 + static_cast<int>(360.f * source[offset])) % 360};

  const float m2 {(l <= 0.5f) ? l * (l + s) : l + s - l * s};

  const float m1 {2 * l - m2};

  optr[offset].x = to_RGB(m1, m2, h + 120);
  optr[offset].y = to_RGB(m1, m2, h);
  optr[offset].z = to_RGB(m1, m2, h - 120);
  optr[offset].w = 255;
}

__global__ void float_to_color_with_linear_saturation(
  unsigned char* optr,
  const float* source)
{
  // Map from threadIdx, BlockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  const float l {source[offset]};
  const float s {linear_float_to_saturation(l)};

  const int h {(180 + static_cast<int>(360.f * source[offset])) % 360};

  const float m2 {(l <= 0.5f) ? l * (l + s) : l + s - l * s};

  const float m1 {2 * l - m2};

  optr[offset * 4 + 0] = to_RGB(m1, m2, h + 120);
  optr[offset * 4 + 1] = to_RGB(m1, m2, h);
  optr[offset * 4 + 2] = to_RGB(m1, m2, h - 120);
  optr[offset * 4 + 3] = 255;
}

__global__ void float_to_color_with_linear_saturation(
  uchar4* optr,
  const float* source)
{
  // Map from threadIdx, BlockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  const float l {source[offset]};
  const float s {linear_float_to_saturation(l)};

  const int h {(180 + static_cast<int>(360.f * source[offset])) % 360};

  const float m2 {(l <= 0.5f) ? l * (l + s) : l + s - l * s};

  const float m1 {2 * l - m2};

  optr[offset].x = to_RGB(m1, m2, h + 120);
  optr[offset].y = to_RGB(m1, m2, h);
  optr[offset].z = to_RGB(m1, m2, h - 120);
  optr[offset].w = 255;
}

} // namespace ColorConversion
} // namespace Visualization