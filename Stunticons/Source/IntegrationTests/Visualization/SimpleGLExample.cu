// needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers
#define GL_GLEXT_PROTOTYPES 

#include "Utilities/HandleUnsuccessfulCudaCall.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectNames.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectParameters.h"

// cudaGraphicsGLRegisterBuffer, cudaGraphicsMapFlagNone
#include <cuda_gl_interop.h> 
#include <cuda_runtime.h> // cudaFree, cudaMalloc, cudaMemcpyAsync
//#include <GL/gl.h> // GLuint

#include <iostream>

using Parameters = Visualization::OpenGLInterface::OpenGLBufferObjectParameters;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::OpenGLInterface::HandleGLError;
using Visualization::OpenGLInterface::OpenGLBufferObjectNames;
using std::cout;

int main()
{
  Parameters parameters {};
  {
    parameters.binding_target_ = GL_PIXEL_UNPACK_BUFFER_ARB;
    parameters.usage_ = GL_DYNAMIC_DRAW_ARB;
  }

  OpenGLBufferObjectNames buffer_object {parameters};

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

  /*
  GLuint buffer_object {};
  cudaGraphicsResource* resource {nullptr};

  cudaGraphicsGLRegisterBuffer(
    &resource,
    buffer_object,
    cudaGraphicsMapFlagsNone);	
  */
}