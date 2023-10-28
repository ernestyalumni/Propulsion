#ifndef VISUALIZATION_FLOAT_TO_COLOR_H
#define VISUALIZATION_FLOAT_TO_COLOR_H

namespace Visualization
{

namespace ColorConversion
{

//------------------------------------------------------------------------------
/// \brief Transforms intermediary products from hsl (hue-saturation-lightness)
/// to rgb (red, green blue)
/// TODO: Determine is there anyway around inlining the implementation of this
/// in header in order to be able to use in a template function.
//------------------------------------------------------------------------------
__device__ unsigned char to_RGB(const float n1, const float n2, const int h);

//------------------------------------------------------------------------------
/// TODO: test this.
/// Provides a simple linear relationship between lightness as an input value
/// and saturation.
//------------------------------------------------------------------------------
inline __device__ float linear_float_to_saturation(const float value)
{
  return 1.0f - fabsf(0.5f - value) * 2.0f;  
}

inline __device__ float float_is_saturation(const float value)
{
  return value;
}

//------------------------------------------------------------------------------
/// TODO: This is considered by the compiler as a host function.
/// Notice the __device__ prefix here. This is important to tell the compiler to
/// run this function and all its overhead on the GPU.
//------------------------------------------------------------------------------
/*
__device__ auto float_is_saturation = [] __device__ (const float value)
{
  return value;
};
*/

// https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768
// https://developer.nvidia.com/blog/new-compiler-features-cuda-8/

// Not needed until we use lambda functions.
/*
#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "Please compile with --expt-extended-lambda"
#endif
*/

__global__ void float_to_color_with_set_saturation(
  unsigned char* optr,
  const float* source);

__global__ void float_to_color_with_set_saturation(
  uchar4* optr,
  const float* source);

__global__ void float_to_color_with_linear_saturation(
  unsigned char* optr,
  const float* source);

__global__ void float_to_color_with_linear_saturation(
  uchar4* optr,
  const float* source);

// TODO: template __global__ functions at runtime seem to make synchronization
// of devices via CUDA fail.

template <typename F>
__global__ void float_to_color(
  unsigned char* optr,
  const float* source,
  F float_to_saturation_function)
{
  // Map from threadIdx, BlockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  const float l {source[offset]};
  const float s {float_to_saturation_function(l)};

  const int h {(180 + static_cast<int>(360.f * source[offset])) % 360};

  const float m2 {(l <= 0.5f) ? l * (l + s) : l + s - l * s};

  const float m1 {2 * l - m2};

  optr[offset * 4 + 0] = to_RGB(m1, m2, h + 120);
  optr[offset * 4 + 1] = to_RGB(m1, m2, h);
  optr[offset * 4 + 2] = to_RGB(m1, m2, h - 120);
  optr[offset * 4 + 3] = 255;
}

template <typename F>
__global__ void float_to_color(
  uchar4* optr,
  const float* source,
  F float_to_saturation_function
    // Notice the __device__ prefix here. This is important to tell the compiler
    // to run this function and all its overhead on the GPU.
    /*
    [] __device__ (const float value)
    {
      return value;
    }
    */
    )
{
  // Map from threadIdx, BlockIdx to pixel position.
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};

  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  const float l {source[offset]};
  const float s {float_to_saturation_function(l)};

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

#endif // VISUALIZATION_FLOAT_TO_COLOR_H