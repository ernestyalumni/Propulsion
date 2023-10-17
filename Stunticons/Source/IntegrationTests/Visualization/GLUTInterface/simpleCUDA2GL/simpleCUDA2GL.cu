namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

//------------------------------------------------------------------------------
/// \brief Clamp or bound a value x between a and b.
//------------------------------------------------------------------------------

__device__ float clamp(const float x, const float a, const float b)
{
  return max(a, min(b, x));
}

__device__ int clamp(const int x, const int a, const int b)
{
  return max(a, min(b, x));
}

//------------------------------------------------------------------------------
/// \brief Convert floating point RGB color to 8-bit integer.
//------------------------------------------------------------------------------
__device__ int rgb_to_int(const float r, const float g, const float b)
{
  static constexpr float lower_bound {0.0f};
  static constexpr float upper_bound {255.0f};

  r = clamp(r, lower_bound, upper_bound);
  g = clamp(r, lower_bound, upper_bound);
  b = clamp(r, lower_bound, upper_bound);

  return (
    static_cast<int>(b) << 16 |
    (static_cast<int>(g) << 8) |
    static_cast<int>(r));
}

__global__ void make_striped_pattern(unsigned int* data, const int image_width)
{
  const int tx {threadIdx.x};
  const int ty {threadIdx.y};
  const int bw {blockIdx.x};
  const int bh {blockIdx.y};
  const int x {blockIdx.x * bw + tx};
  const int y {blockIdx.y * bh + ty};

  uchar4 c4 {make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0)};

  data[y * image_width + x] = rgb_to_int(c4.z, c4.y, c4.x);
}

void make_striped_pattern(
  const dim3 grid,
  const dim3 block,
  unsigned int* data,
  const int image_width)
{
  make_striped_pattern<<<grid, block>>>(data, image_width);
}

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests