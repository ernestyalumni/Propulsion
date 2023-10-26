#ifndef VISUALIZATION_GLFW_INTERFACE_MAKE_BITMAP_H
#define VISUALIZATION_GLFW_INTERFACE_MAKE_BITMAP_H

#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLFWInterface/GLFWWindow.h"
#include "Visualization/MappedDevicePointer.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "Visualization/OpenGLInterface/BufferObjectParameters.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/DrawPixels.h"

#include <GLFW/glfw3.h>
#include <cstddef>
#include <string>

namespace Visualization
{
namespace GLFWInterface
{

class MakeBitMap
{
  public:

    using BufferObjectNames = Visualization::OpenGLInterface::BufferObjectNames;
    using BufferObjectParameters =
      Visualization::OpenGLInterface::BufferObjectParameters;
    using CUDAGraphicsResource = Visualization::CUDAGraphicsResource;
    using CreateOpenGLBuffer =
      Visualization::OpenGLInterface::CreateOpenGLBuffer;
    using DrawPixels = Visualization::OpenGLInterface::DrawPixels;
    using GLFWWindowParameters = GLFWWindow::Parameters;
    template <typename T>
    using MappedDevicePointer = Visualization::MappedDevicePointer<T>;

    inline static BufferObjectParameters make_default_buffer_object_parameters(
      const std::size_t width,
      const std::size_t height)
    {
      return BufferObjectParameters{
        1,
        GL_PIXEL_UNPACK_BUFFER_ARB,
        GL_DYNAMIC_DRAW_ARB,
        width,
        height};
    }

    inline static DrawPixels::Parameters make_default_draw_pixels_parameters(
      const std::size_t width,
      const std::size_t height)
    {
      return DrawPixels::Parameters {width, height, GL_RGBA, GL_UNSIGNED_BYTE};
    }

    MakeBitMap(
      const GLFWWindowParameters& glfw_window_parameters,
      const BufferObjectParameters& buffer_parameters,
      const CUDAGraphicsResource::Parameters& cuda_graphics_parameters,
      const DrawPixels::Parameters& draw_pixels_parameters
      ):
      glfw_window_parameters_{glfw_window_parameters},
      buffer_parameters_{buffer_parameters},
      cuda_graphics_parameters_{cuda_graphics_parameters},
      draw_pixels_parameters_{draw_pixels_parameters}
    {}

    template <typename F>
    bool run(F render_object)
    {
      bool no_errors {true};

      GLFWWindow glfw_window {};

      glfw_window.initialize();

      no_errors &= glfw_window.create_window(glfw_window_parameters_);      

      BufferObjectNames buffer_object {buffer_parameters_};
      buffer_object.initialize();

      CreateOpenGLBuffer create_buffer {};
      no_errors &= create_buffer.create_buffer_object_data(buffer_parameters_);

      CUDAGraphicsResource cuda_graphics_resource {};
      const auto register_result = cuda_graphics_resource.register_buffer_object(
        cuda_graphics_parameters_,
        buffer_object);
      no_errors &= register_result.is_cuda_success();

      const auto map_result = cuda_graphics_resource.map_resource();
      no_errors &= map_result.is_cuda_success();

      MappedDevicePointer<uchar4> mapped_device_pointer {};
      mapped_device_pointer.get_mapped_device_pointer(cuda_graphics_resource);

      render_object.run(mapped_device_pointer.device_pointer_);

      const auto unmap_result = cuda_graphics_resource.unmap_resource();
      no_errors &= unmap_result.is_cuda_success();

      while (!glfwWindowShouldClose(glfw_window.created_window_handle_))
      {
        const auto draw_result = DrawPixels::draw_pixels_to_frame_buffer(
          draw_pixels_parameters_);

        no_errors &= draw_result.is_no_gl_error();

        glfwSwapBuffers(glfw_window.created_window_handle_);
        glfwPollEvents();        
      }

      return no_errors;
    }

  private:

    GLFWWindowParameters glfw_window_parameters_;
    BufferObjectParameters buffer_parameters_;
    CUDAGraphicsResource::Parameters cuda_graphics_parameters_;
    DrawPixels::Parameters draw_pixels_parameters_;
};

MakeBitMap create_make_bit_map(
  const std::size_t width,
  const std::size_t height);

} // namespace GLFWInterface
} // namespace Visualization

#endif // VISUALIZATION_GLFW_INTERFACE_MAKE_BITMAP_H
