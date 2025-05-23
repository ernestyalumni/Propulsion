// needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers
#define GL_GLEXT_PROTOTYPES 

#include "Utilities/HandleUnsuccessfulCUDACall.h"
#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLUTInterface/GLUTWindow.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "Visualization/OpenGLInterface/BufferObjectParameters.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"

// cudaGraphicsGLRegisterBuffer, cudaGraphicsMapFlagNone
#include <cuda_gl_interop.h> 
#include <cuda_runtime.h> // cudaFree, cudaMalloc, cudaMemcpyAsync
//#include <GL/gl.h> // GLuint
#include <GL/glut.h> // glutInit

#include <iostream>

using Parameters = Visualization::OpenGLInterface::BufferObjectParameters;
using Utilities::HandleUnsuccessfulCUDACall;
using Visualization::CUDAGraphicsResource;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::OpenGLInterface::HandleGLError;
using Visualization::OpenGLInterface::BufferObjectNames;
using std::cout;

int main(int argc, char **argv)
{
  Visualization::GLUTInterface::GLUTWindow::Parameters glut_parameters {
    {GLUT_DOUBLE | GLUT_RGBA}};
  glut_parameters.display_name_ = "bitmap";
  glut_parameters.width_ = 512;
  glut_parameters.height_ = 512;

  Visualization::GLUTInterface::GLUTWindow::instance().initialize_glut(
    &argc,
    argv,
    glut_parameters);
  /*
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(512, 512);
  glutCreateWindow("bitmap");
  */

  HandleGLError gl_err {};

  // Uncomment to explicitly running the underlying steps as a sanity check.
  /*
  GLuint buffer_object {0};
  struct cudaGraphicsResource* cuda_graphics_resource {};
  glGenBuffers(1, &buffer_object);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_object);
  glBufferData(
    GL_PIXEL_UNPACK_BUFFER_ARB,
    512 * 512 * 4 * sizeof(GLubyte),
    nullptr,
    GL_STREAM_DRAW_ARB);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  */

  Parameters parameters {};
  {
    parameters.binding_target_ = GL_PIXEL_UNPACK_BUFFER_ARB;
    parameters.usage_ = GL_DYNAMIC_DRAW_ARB;
  }

  BufferObjectNames buffer_object {parameters};

  cout << (parameters.number_of_buffer_object_names_ == 1) << "\n";
  cout << (buffer_object.buffer_objects_ == nullptr) << "\n";

  // Sanity check where we explicitly run the steps for initializing an OpenGL 
  // buffer object name.
  /*
  glGenBuffers(
    parameters.number_of_buffer_object_names_,
    &(buffer_object.buffer_object_));

  glBindBuffer(parameters.binding_target_, buffer_object.buffer_object_);
  */

  cout << buffer_object.initialize() << "\n";

  CreateOpenGLBuffer create_buffer {};

  cout << create_buffer.create_buffer_object_data(parameters) << "\n";

  gl_err();
  cout << gl_err.is_no_gl_error() << "\n";

  HandleUnsuccessfulCUDACall handle_register {
    "Failed to register OpenGL buffer object"};

  // Uncomment out and comment out respective part to try different calls.
  /*
  handle_register(cudaGraphicsGLRegisterBuffer(
    &cuda_graphics_resource,
    buffer_object,
    cudaGraphicsMapFlagsWriteDiscard));
  */
  /*
  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_register,
    cudaGraphicsGLRegisterBuffer(
      &cuda_graphics_resource,
      buffer_object,
      cudaGraphicsMapFlagsWriteDiscard));
  */

  CUDAGraphicsResource cuda_graphics_resource {};
  const CUDAGraphicsResource::Parameters cuda_parameters {};
  // Uncomment this and comment the next call to try this option.
  const auto handle_call = cuda_graphics_resource.register_buffer_object(
    cuda_parameters,
    buffer_object);

  cout << handle_call.is_cuda_success() << "\n";
}