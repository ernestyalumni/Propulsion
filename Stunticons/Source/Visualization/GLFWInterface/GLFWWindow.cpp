#include "Visualization/GLFWInterface/GLFWWindow.h"

#include <GLFW/glfw3.h>

namespace Visualization
{
namespace GLFWInterface
{

bool GLFWWindow::create_window(const Parameters& parameters)
{
  created_window_handle_ = glfwCreateWindow(
    parameters.width_,
    parameters.height_,
    parameters.display_name_.data(),
    nullptr,
    share_);

  glfwMakeContextCurrent(created_window_handle_);

  return is_window_created();
}

} // namespace GLFWInterface
} // namespace Visualization
