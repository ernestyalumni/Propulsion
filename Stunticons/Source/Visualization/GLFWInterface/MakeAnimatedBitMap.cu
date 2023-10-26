#include "Visualization/GLFWInterface/MakeAnimatedBitMap.h"

#include <cstddef>

namespace Visualization
{
namespace GLFWInterface
{

MakeAnimatedBitMap create_make_animated_bit_map(
  const std::size_t width,
  const std::size_t height,
  const std::string window_title)
{
  const auto buffer_object_parameters =
    MakeBitMap::make_default_buffer_object_parameters(width, height);

  const auto draw_pixels_parameters =
    MakeBitMap::make_default_draw_pixels_parameters(width, height);

  const MakeAnimatedBitMap::GLFWWindowParameters glfw_window_parameters {
    window_title,
    width,
    height};

  return MakeAnimatedBitMap {
    glfw_window_parameters,
    buffer_object_parameters,
    MakeBitMap::CUDAGraphicsResource::Parameters {},
    draw_pixels_parameters};
}

} // namespace GLFWInterface
} // namespace Visualization
