#include "float_to_color.h"

namespace Visualization
{
namespace ColorConversion
{

__device__ unsigned char to_rgb(const float n1, const float n2, const int hue)
{
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

} // namespace ColorConversion
} // namespace Visualization