#include "JuliaSet.h"

#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/IsInJuliaSet.h"
#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/Parameters.h"
#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLUTInterface/GLUTWindow.h"
#include "Visualization/MappedDevicePointer.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/DrawPixels.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"

#include <GL/glut.h> // GLUT_RGBA
#include <cuda_runtime.h>
#include <functional>

using IntegrationTests::Visualization::GLUTInterface::JuliaSet::
  get_default_julia_parameters;

using GLUTWindowParameters =
  Visualization::GLUTInterface::GLUTWindow::Parameters;
using BufferObjectParameters =
  Visualization::OpenGLInterface::BufferObjectNames::Parameters;
using Visualization::CUDAGraphicsResource;
using Visualization::MappedDevicePointer;
using Visualization::OpenGLInterface::BufferObjectNames;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::OpenGLInterface::DrawPixels;
using Visualization::OpenGLInterface::HandleGLError;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{
namespace JuliaSet
{

JuliaSet::Parameters::Parameters(
  const unsigned int image_width,
  const unsigned int image_height,
  const dim3 blocks
  ):
  image_width_{image_width},
  image_height_{image_height},
  blocks_{blocks},
  threads_{image_width / blocks.x, image_height / blocks.y, 1}
{}

const JuliaSet::Parameters JuliaSet::default_parameters_{
  dimensions_,
  dimensions_,
  1};

const GLUTWindowParameters
  JuliaSet::default_glut_window_parameters_{
    "Julia Sets bit map",
    dimensions_,
    dimensions_,
    {GLUT_DOUBLE, GLUT_RGBA}};

void JuliaSet::draw_function()
{
  DrawPixels::draw_pixels_to_frame_buffer(
    DrawPixels::Parameters {
      JuliaSet::dimensions_,
      JuliaSet::dimensions_,
      GL_RGBA,
      GL_UNSIGNED_BYTE
    });

  DrawPixels::swap_buffers();
}

bool JuliaSet::run(int* argcp, char** argv)
{
  bool no_error {true};

  float scale {1.5};

  // Start of "initGL"

  ::Visualization::GLUTInterface::GLUTWindow::instance().initialize_glut(
    argcp,
    argv,
    default_glut_window_parameters_);

  HandleGLError gl_err {};

  // End of "initGL"

  BufferObjectParameters buffer_parameters {};
  buffer_parameters.binding_target_ = GL_PIXEL_UNPACK_BUFFER_ARB;
  buffer_parameters.usage_ = GL_DYNAMIC_DRAW_ARB;
  buffer_parameters.width_ = dimensions_;
  buffer_parameters.height_ = dimensions_;

  BufferObjectNames buffer_object {buffer_parameters};
  buffer_object.initialize();
  
  CreateOpenGLBuffer create_buffer {};
  no_error &= create_buffer.create_buffer_object_data(buffer_parameters);

  CUDAGraphicsResource cuda_graphics_resource {};
  const CUDAGraphicsResource::Parameters cuda_parameters {};
  cuda_graphics_resource.register_buffer_object(
    cuda_parameters,
    buffer_object);

  cuda_graphics_resource.map_resource();

  MappedDevicePointer<uchar4> mapped_device_pointer {};
  mapped_device_pointer.get_mapped_device_pointer(cuda_graphics_resource);

  is_in_julia_set<<<parameters_.threads_, parameters_.blocks_>>>(
    mapped_device_pointer.device_pointer_,
    scale,
    get_default_julia_parameters(dimensions_, dimensions_));

  return no_error;
}

void JuliaSet::display_and_exit(CUDAGraphicsResource& cuda_graphics_resource)
{
  cuda_graphics_resource.unmap_resource();

  glutDisplayFunc(draw_function);
  glutMainLoop();
}

} // namespace JuliaSet
} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests