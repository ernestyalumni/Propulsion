#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_JULIA_SET_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_JULIA_SET_H

#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLUTInterface/GLUTWindow.h"

#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{
namespace JuliaSet
{

class JuliaSet
{
  public:

    inline static constexpr int dimensions_ {1500};

    using GLUTWindowParameters =
      ::Visualization::GLUTInterface::GLUTWindow::Parameters;

    struct Parameters
    {
      Parameters(
        const unsigned int image_width,
        const unsigned int image_height,
        const dim3 blocks);

      unsigned int image_width_;
      unsigned int image_height_;
      dim3 blocks_;
      dim3 threads_;
    };

    static const Parameters default_parameters_;

    static const GLUTWindowParameters default_glut_window_parameters_;

    JuliaSet(const Parameters& parameters = default_parameters_):
      parameters_{parameters}
    {}

    //--------------------------------------------------------------------------
    /// TODO: GLUT is not working properly.
    //--------------------------------------------------------------------------
    bool run(int* argcp, char** argv);

    bool run_with_GLFW(int* argcp, char** argv);

  private:

    static void draw_function();

    void display_and_exit(
      ::Visualization::CUDAGraphicsResource& cuda_graphics_resource);

    Parameters parameters_;
};

} // namespace JuliaSet
} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_JULIA_SET_H