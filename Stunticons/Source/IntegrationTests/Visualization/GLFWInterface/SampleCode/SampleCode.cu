#include "IntegrationTests/Visualization/GLFWInterface/SampleCode/SampleCode.h"

#include "DataStructures/Array.h"
#include "IntegrationTests/Visualization/GLFWInterface/SampleCode/fill_RGB.h"
#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLFWInterface/GLFWWindow.h"
#include "Visualization/MappedDevicePointer.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/DrawPixels.h"

#include <algorithm>
#include <cstddef>
#include <vector>

using GLFWWindowParameters =
  Visualization::GLFWInterface::GLFWWindow::Parameters;

using BufferObjectParameters =
  Visualization::OpenGLInterface::BufferObjectNames::Parameters;

using DataStructures::Array;
using IntegrationTests::Visualization::GLFWInterface::SampleCode::fill_RGB;
using Visualization::CUDAGraphicsResource;
using Visualization::GLFWInterface::GLFWWindow;
using Visualization::MappedDevicePointer;
using Visualization::OpenGLInterface::BufferObjectNames;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::OpenGLInterface::DrawPixels;
using std::size_t;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SampleCode
{

SampleCode::Parameters::Parameters(
  const size_t window_width,
  const size_t window_height,
  const size_t threads_per_block
  ):
  window_width_{window_width},
  window_height_{window_height},
  blocks_per_grid_{
    static_cast<unsigned int>(
      (window_height * window_width + threads_per_block - 1) /
        threads_per_block)},
  threads_per_block_{static_cast<unsigned int>(threads_per_block)}
{}

const SampleCode::Parameters SampleCode::default_test_sample_parameters_{
  SampleCode::width_,
  SampleCode::height_,
  SampleCode::threads_per_block_};

const GLFWWindowParameters
  SampleCode::default_glfw_window_parameters_{
    "Custom RGB Window",
    SampleCode::width_,
    SampleCode::height_};

bool SampleCode::run(int* argcp, char** argv)
{
  bool no_errors {true};

  GLFWWindow glfw_window {};
  glfw_window.initialize();

  no_errors |= glfw_window.create_window(
    SampleCode::default_glfw_window_parameters_);

  BufferObjectParameters buffer_parameters {};
  buffer_parameters.binding_target_ = GL_PIXEL_UNPACK_BUFFER_ARB;
  buffer_parameters.usage_ = GL_DYNAMIC_DRAW_ARB;
  buffer_parameters.width_ = 800;
  buffer_parameters.height_ = 600;

  BufferObjectNames buffer_object {buffer_parameters};
  buffer_object.initialize();
  
  CreateOpenGLBuffer create_buffer {};
  no_errors |= create_buffer.create_buffer_object_data(buffer_parameters);

  CUDAGraphicsResource cuda_graphics_resource {};
  const CUDAGraphicsResource::Parameters cuda_parameters {};
  cuda_graphics_resource.register_buffer_object(
    cuda_parameters,
    buffer_object);

  return no_errors;
}

bool SampleCode::run_sample_code(int* argcp, char** argv)
{
  bool no_errors {true};

  GLFWWindow glfw_window {};
  glfw_window.initialize();

  no_errors |= glfw_window.create_window(
    SampleCode::default_glfw_window_parameters_);

  Array<unsigned char> array {
    SampleCode::default_glfw_window_parameters_.width_ *
      SampleCode::default_glfw_window_parameters_.height_ *
      3};

  //fill_RGB<<<parameters_.blocks_per_grid_, parameters_.threads_per_block_>>>(
  //  array.elements_);

  const size_t threads_per_block {256};
  const size_t blocks_per_grid {
    (width_ * height_ + threads_per_block - 1) / threads_per_block};
  fill_RGB<<<blocks_per_grid, threads_per_block>>>(array.elements_);

  // Allocate memory for RGB buffer on host
  unsigned char* host_rgb {new unsigned char[width_ * height_ * 3]};

  array.copy_device_output_to_host(host_rgb);

  // TODO: See if this can be fixed.
  /*
  std::vector<unsigned char> host_vec_rgb (
    SampleCode::default_glfw_window_parameters_.width_ *
      SampleCode::default_glfw_window_parameters_.height_ *
        3);

  array.copy_device_output_to_host(host_vec_rgb);
  */

  // Loop until the user closes the window.
  while (!glfwWindowShouldClose(glfw_window.created_window_handle_))
  {
    // Render RGB Buffer to window
    glClear(GL_COLOR_BUFFER_BIT);

    /*
    DrawPixels::draw_pixels_to_frame_buffer(
      DrawPixels::Parameters {
        SampleCode::width_,
        SampleCode::height_,
        GL_RGBA,
        GL_UNSIGNED_BYTE
      },
      host_rgb);
    */
    glDrawPixels(width_, height_, GL_RGB, GL_UNSIGNED_BYTE, host_rgb);

    glfwSwapBuffers(glfw_window.created_window_handle_);
    glfwPollEvents();
  }

  delete[] host_rgb;
  return no_errors;
}

} // namespace SampleCode
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests