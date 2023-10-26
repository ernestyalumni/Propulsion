#include "Visualization/GLFWInterface/MakeBitMap.h"

#include <cstddef>

namespace Visualization
{
namespace GLFWInterface
{

MakeBitMap create_make_bit_map(
  const std::size_t width,
  const std::size_t height,
  const std::string window_title)
{
  const auto buffer_object_parameters =
    MakeBitMap::make_default_buffer_object_parameters(width, height);

  const auto draw_pixels_parameters =
    MakeBitMap::make_default_draw_pixels_parameters(width, height);

  const MakeBitMap::GLFWWindowParameters glfw_window_parameters {
    window_title,
    width,
    height};

  return MakeBitMap {
    glfw_window_parameters,
    buffer_object_parameters,
    MakeBitMap::CUDAGraphicsResource::Parameters {},
    draw_pixels_parameters};
}

} // namespace GLFWInterface
} // namespace Visualization
