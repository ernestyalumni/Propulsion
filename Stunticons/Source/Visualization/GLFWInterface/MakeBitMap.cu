#include "Visualization/GLFWInterface/MakeBitMap.h"

#include <cstddef>

namespace Visualization
{
namespace GLFWInterface
{

void MakeBitMap::keyboard_callback(
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
