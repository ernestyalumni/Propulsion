#include "IntegrationTests/Visualization/GLFWInterface/SampleCode/SampleCode.h"

#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLFWInterface/GLFWWindow.h"
#include "Visualization/MappedDevicePointer.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"

using GLFWWindowParameters =
  Visualization::GLFWInterface::GLFWWindow::Parameters;

using BufferObjectParameters =
  Visualization::OpenGLInterface::BufferObjectNames::Parameters;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::CUDAGraphicsResource;
using Visualization::GLFWInterface::GLFWWindow;
using Visualization::MappedDevicePointer;
using Visualization::OpenGLInterface::BufferObjectNames;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SampleCode
{

const GLFWWindowParameters
  SampleCode::default_glfw_window_parameters_{
    "Custom RGB Window",
    800,
    600};

bool SampleCode::run(int* argcp, char** argv)
{
  bool no_errors {true};

  GLFWWindow glfw_window {};
  glfw_window.initialize_library();

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

} // namespace SampleCode
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests