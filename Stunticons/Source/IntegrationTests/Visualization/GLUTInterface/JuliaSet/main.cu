// Previously used for sanity check.
// #include "IntegrationTests/Visualization/GLUTInterface/GPUBitMap.h"

#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/IsInJuliaSet.h"
#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/JuliaSet.h"
#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLUTInterface/GLUTWindow.h"
#include "Visualization/MappedDevicePointer.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"

#include <GL/glut.h> // GLUT_RGBA
#include <stdexcept>

using IntegrationTests::Visualization::GLUTInterface::JuliaSet::
  get_default_julia_parameters;

using BufferObjectParameters =
  Visualization::OpenGLInterface::BufferObjectNames::Parameters;

using IntegrationTests::Visualization::GLUTInterface::JuliaSet::JuliaSet;
using Visualization::CUDAGraphicsResource;
using Visualization::MappedDevicePointer;
using Visualization::OpenGLInterface::BufferObjectNames;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;

BufferObjectParameters buffer_parameters {
  1,
  GL_PIXEL_UNPACK_BUFFER_ARB,
  GL_DYNAMIC_DRAW_ARB,
  JuliaSet::dimensions_,
  JuliaSet::dimensions_};

BufferObjectNames buffer_object {buffer_parameters};

CUDAGraphicsResource cuda_graphics_resource {};

void draw_function()
{
  glDrawPixels(
    JuliaSet::dimensions_,
    JuliaSet::dimensions_,
    GL_RGBA,GL_UNSIGNED_BYTE,
    0);

  glutSwapBuffers();
}

void keyboard_callback(const unsigned char key, int, int)
{
  switch (key)
  {
    case 27:
      cuda_graphics_resource.unregister_buffer_object();
      buffer_object.delete_buffer_objects();
      exit(0);
  }
}

int main(int argc, char* argv[])
{
  float scale {1.5};

  // Previous sanity check.
  /*
  GPUBitmap bitmap {JuliaSet::dimensions_, JuliaSet::dimensions_};

  is_in_julia_set<<<
    JuliaSet::default_parameters_.threads_,
    JuliaSet::default_parameters_.blocks_>>>(
      bitmap.devPtr,
      scale,
      get_default_julia_parameters(
        JuliaSet::dimensions_,
        JuliaSet::dimensions_));

  bitmap.display_and_exit();
  */

  // TODO: fix Failed to unregister CUDA resource (error code driver shutting down)!
  ::Visualization::GLUTInterface::GLUTWindow::instance().initialize_glut(
    &argc,
    argv,
    JuliaSet::default_glut_window_parameters_);

  buffer_object.initialize();

  CreateOpenGLBuffer create_buffer {};
  create_buffer.create_buffer_object_data(buffer_parameters);

  const CUDAGraphicsResource::Parameters cuda_parameters {};
  const auto register_result = cuda_graphics_resource.register_buffer_object(
    cuda_parameters,
    buffer_object);

  if (!register_result.is_cuda_success())
  {
    throw std::runtime_error("Register CUDA Graphics Resource failed.");
  }

  const auto map_result = cuda_graphics_resource.map_resource();

  if (!map_result.is_cuda_success())
  {
    throw std::runtime_error("Map CUDA Graphics Resource failed.");
  }

  MappedDevicePointer<uchar4> mapped_device_pointer {};
  mapped_device_pointer.get_mapped_device_pointer(cuda_graphics_resource);

  is_in_julia_set<<<
    JuliaSet::default_parameters_.threads_,
    JuliaSet::default_parameters_.blocks_>>>(
      mapped_device_pointer.device_pointer_,
      scale,
      get_default_julia_parameters(
        JuliaSet::dimensions_,
        JuliaSet::dimensions_));

  cuda_graphics_resource.unmap_resource();

  glutDisplayFunc(draw_function);
  glutKeyboardFunc(keyboard_callback);

  glutMainLoop();

  // Not ready yet.
  /*
  JuliaSet julia_set {};
  julia_set.run(&argc, argv);
  */
}