#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_SAMPLE_CODE_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_SAMPLE_CODE_H

#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLFWInterface/GLFWWindow.h"

#include <cstddef>
#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SampleCode
{

class SampleCode
{
  public:

    using GLFWWindowParameters =
      ::Visualization::GLFWInterface::GLFWWindow::Parameters;

    inline static constexpr std::size_t width_ {800};
    inline static constexpr std::size_t height_ {600};
    inline static constexpr std::size_t threads_per_block_ {256};

    struct Parameters
    {
      Parameters(
        const std::size_t window_width,
        const std::size_t window_height,
        const std::size_t threads_per_block);

      std::size_t window_width_;
      std::size_t window_height_;
      dim3 blocks_per_grid_;
      dim3 threads_per_block_;
    };

    static const Parameters default_test_sample_parameters_;

    static const GLFWWindowParameters default_glfw_window_parameters_;

    SampleCode(const Parameters& parameters = default_test_sample_parameters_):
      parameters_{parameters}
    {}

    bool run(int* argcp, char** argv);

    //--------------------------------------------------------------------------
    /// \details This sample is not to demonstrate best practices for
    /// production at all as the cudaMemcpy calls are expensive, but as a test
    /// of GLFW and CUDA. 
    //--------------------------------------------------------------------------
    bool run_sample_code(int* argcp, char** argv);

  private:

    Parameters parameters_;
};

} // namespace SampleCode
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_SAMPLE_CODE_H