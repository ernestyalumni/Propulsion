#include "IntegrationTests/Visualization/GLFWInterface/Ripple/Ripple.h"

#include "IntegrationTests/Visualization/GLFWInterface/Ripple/calculate_ripple_value.h"
#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLFWInterface/GLFWWindow.h"
#include "Visualization/MappedDevicePointer.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/DrawPixels.h"

using BufferObjectParameters =
  Visualization::OpenGLInterface::BufferObjectNames::Parameters;
using RippleParameters =
  IntegrationTests::Visualization::GLFWInterface::Ripple::Parameters;

using
  IntegrationTests::Visualization::GLFWInterface::Ripple::generate_frame;
using
  IntegrationTests::Visualization::GLFWInterface::Ripple::
    calculate_ripple_values;
using Visualization::CUDAGraphicsResource;
using Visualization::GLFWInterface::GLFWWindow;
using Visualization::MappedDevicePointer;
using Visualization::OpenGLInterface::BufferObjectNames;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::OpenGLInterface::DrawPixels;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace Ripple
{

void render(
  CUDAGraphicsResource& cuda_graphics_resource,
  RippleParameters ripple_parameters,
  const int ticks)
{
  cuda_graphics_resource.map_resource();

  MappedDevicePointer<uchar4> mapped_device_pointer {};
  mapped_device_pointer.get_mapped_device_pointer(cuda_graphics_resource);

  // This works, this was a sanity check using the original code.
  /*
  generate_frame(
    mapped_device_pointer.device_pointer_,
    nullptr,
    ticks);
  */

  calculate_ripple_values(
    mapped_device_pointer.device_pointer_,
    ticks,
    ripple_parameters);
 
  cuda_graphics_resource.unmap_resource();
}

void Ripple::run(int* argcp, char** argv)
{
  GLFWWindow glfw_window {};
  glfw_window.initialize();

  glfw_window.create_window(
    Ripple::GLFWWindowParameters{
      "GPU Ripple animation",
      Ripple::dimension_,
      Ripple::dimension_});

  BufferObjectParameters buffer_parameters {};
  buffer_parameters.binding_target_ = GL_PIXEL_UNPACK_BUFFER_ARB;
  buffer_parameters.usage_ = GL_DYNAMIC_DRAW_ARB;
  buffer_parameters.width_ = Ripple::dimension_;
  buffer_parameters.height_ = Ripple::dimension_;

  BufferObjectNames buffer_object {buffer_parameters};
  buffer_object.initialize();
  
  CreateOpenGLBuffer create_buffer {};
  create_buffer.create_buffer_object_data(buffer_parameters);

  CUDAGraphicsResource cuda_graphics_resource {};
  const CUDAGraphicsResource::Parameters cuda_parameters {};
  cuda_graphics_resource.register_buffer_object(
    cuda_parameters,
    buffer_object);

  int ticks {0};

  while (!glfwWindowShouldClose(glfw_window.created_window_handle_))
  {
    render(
      cuda_graphics_resource,
      RippleParameters{Ripple::dimension_, Ripple::threads_per_block_},
      ticks++);

    //--------------------------------------------------------------------------
    /// TODO: determine if we should include the following 2 lines.
    /// The following 2 lines don't seem necessary for successfuly render.
    /// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glClearColor.xhtml
    /// glClearColor - specify clear values for color buffers.
    /// Specify red, green, blue, alpha values when color buffers are cleared.
    /// Initial values are all 0. glClearColor are clamped to range [0, 1].
    ///
    /// https://registry.khronos.org/OpenGL-Refpages/gl4/html/glClear.xhtml
    /// glClear - clear buffers to preset values
    //--------------------------------------------------------------------------
    /*
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    */

    DrawPixels::draw_pixels_to_frame_buffer(
      DrawPixels::Parameters {
        Ripple::dimension_,
        Ripple::dimension_,
        GL_RGBA,
        GL_UNSIGNED_BYTE});

    glfwSwapBuffers(glfw_window.created_window_handle_);
    glfwPollEvents();
  }
}

} // namespace Ripple
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests