#include "calculate_ripple_value.h"

#include <cstddef>

using std::size_t;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace Ripple
{

__global__ void calculate_ripple_value(
  uchar4* ptr,
  const int ticks,
  const Parameters parameters)
{
  const unsigned int x {threadIdx.x + blockIdx.x * blockDim.x};
  const unsigned int y {threadIdx.y + blockIdx.y * blockDim.y};
  const unsigned int offset {x + y * blockDim.x * gridDim.x};

  // Now calculate the value at that position.
  const float fx {
    static_cast<float>(static_cast<int>(x) - parameters.dimension_ / 2)};
  const float fy {
    static_cast<float>(static_cast<int>(y) - parameters.dimension_ / 2)};
  const float d {sqrtf(fx * fx + fy * fy)};

  const unsigned char grey {
    static_cast<unsigned char>(
      128.0f + 127.0f * cosf(d / 10.0f - ticks / 7.0f) /
        (d / 10.0f + 1.0f))};

  ptr[offset].x = grey;
  ptr[offset].y = grey;
  ptr[offset].z = grey;
  ptr[offset].w = 255;
}

void calculate_ripple_values(
  uchar4* pixels,
  const int ticks,
  const Parameters parameters)
{
  const dim3 threads_per_block {
    static_cast<unsigned int>(parameters.number_of_threads_),
    static_cast<unsigned int>(parameters.number_of_threads_)};

  calculate_ripple_value<<<parameters.blocks_per_grid_, threads_per_block>>>(
    pixels,
    ticks,
    parameters);
}

//------------------------------------------------------------------------------
/// Original code
//------------------------------------------------------------------------------

__global__ void kernel(uchar4* ptr, int ticks )
{
  static constexpr int DIM {2048};

  // map from threadIdx/blockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  
  // now calculate the value at that position
  float fx = x - DIM/2;
  float fy = y - DIM/2;
  float d = sqrt(fx * fx + fy * fy );
  unsigned char grey = (unsigned char) (128.0f + 127.0f * 
    cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f ) );
  
  ptr[offset].x = grey;
  ptr[offset].y = grey;
  ptr[offset].z = grey;
  ptr[offset].w = 255;
} 

void generate_frame(uchar4 *pixels, void*, int ticks ) {
  static constexpr int DIM {2048};

  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16,16);
  kernel<<<grids,threads>>>(pixels, ticks);
}

} // namespace Ripple
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests
