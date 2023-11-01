#include "Visualization/GLFWInterface/MakeAnimatedBitMap.h"

#include <cstddef>

namespace Visualization
{
namespace GLFWInterface
{

void MakeAnimatedBitMap::keyboard_callback(
  GLFWwindow* window_handle,
  int key,
  int scancode,
  int action,
  int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
  {
    // https://www.glfw.org/docs/3.3/group__window.html#ga49c449dde2a6f87d996f4daaa09d6708
    // void glfwSetWindowShouldClose(GLFWwindow* window, int value)
    // This function sets value of the close flag of specified window. This can
    // be used to close the window, or to signal that it should be closed.
    // [in] value - The new value.
    // https://www.glfw.org/docs/3.3/group__init.html
    // GLFW_TRUE is defined as a macro of value 1.
    glfwSetWindowShouldClose(window_handle, GLFW_TRUE);
  }
}

MakeAnimatedBitMap create_make_animated_bit_map(
  const std::size_t width,
  const std::size_t height,
  const std::string window_title)
{
  const auto buffer_object_parameters =
    MakeAnimatedBitMap::make_default_buffer_object_parameters(width, height);

  const auto draw_pixels_parameters =
    MakeAnimatedBitMap::make_default_draw_pixels_parameters(width, height);

  const MakeAnimatedBitMap::GLFWWindowParameters glfw_window_parameters {
    window_title,
    width,
    height};

  return MakeAnimatedBitMap {
    glfw_window_parameters,
    buffer_object_parameters,
    MakeAnimatedBitMap::CUDAGraphicsResource::Parameters {},
    draw_pixels_parameters};
}

} // namespace GLFWInterface
} // namespace Visualization
