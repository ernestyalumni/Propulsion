#ifndef VISUALIZATION_GLFW_INTERFACE_GLFW_WINDOW_H
#define VISUALIZATION_GLFW_INTERFACE_GLFW_WINDOW_H

#include <GLFW/glfw3.h>
#include <cstddef>
#include <string>

namespace Visualization
{
namespace GLFWInterface
{

class GLFWWindow
{
  public:

    struct Parameters
    {
      std::string display_name_;

      std::size_t width_;
      std::size_t height_;
    };

    GLFWWindow():
      share_{nullptr},
      created_window_handle_{nullptr},
      is_initialized_{false}
    {}

    //--------------------------------------------------------------------------
    /// https://www.glfw.org/docs/3.3/group__init.html#ga317aac130a235ab08c6db0834907d85e
    //--------------------------------------------------------------------------
    inline static bool initialize_library()
    {
      return (glfwInit() == GLFW_TRUE);
    }

    //--------------------------------------------------------------------------
    /// https://www.glfw.org/docs/3.3/intro_guide.html#intro_init
    /// Before application exits, you should terminate GLFW library, if it has
    /// been initialized.
    //--------------------------------------------------------------------------
    inline static void terminate_library()
    {
      glfwTerminate();
    }

    bool create_window(const Parameters& parameters);

    inline bool is_window_created()
    {
      return created_window_handle_ != nullptr;
    }

    GLFWwindow* share_;
    GLFWwindow* created_window_handle_;

  private:

    bool is_initialized_;
};

} // namespace GLFWInterface
} // namespace Visualization

#endif // VISUALIZATION_GLFW_INTERFACE_GLFW_WINDOW_H
