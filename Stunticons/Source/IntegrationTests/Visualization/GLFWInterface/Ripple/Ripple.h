#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_RIPPLE_RIPPLE_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_RIPPLE_RIPPLE_H

#include "Visualization/GLFWInterface/GLFWWindow.h"

#include <cstddef>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace Ripple
{

class Ripple
{
  public:

    using GLFWWindowParameters =
      ::Visualization::GLFWInterface::GLFWWindow::Parameters;

    inline static constexpr std::size_t dimension_ {2048};
    inline static constexpr std::size_t threads_per_block_ {16};

    static void run(int* argcp, char** argv);
};

} // namespace Ripple
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_RIPPLE_RIPPLE_H