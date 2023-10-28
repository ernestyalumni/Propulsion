#ifndef VISUALIZATION_FLOAT_TO_COLOR_H
#define VISUALIZATION_FLOAT_TO_COLOR_H

namespace Visualization
{

namespace ColorConversion
{

//------------------------------------------------------------------------------
/// \brief Transforms intermediary products from hsl (hue-saturation-lightness)
/// to rgb (red, green blue)
//------------------------------------------------------------------------------
__device__ unsigned char to_rgb(const float n1, const float n2, const int hue);

} // namespace ColorConversion
} // namespace Visualization

#endif // VISUALIZATION_FLOAT_TO_COLOR_H