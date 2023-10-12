// needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers
#define GL_GLEXT_PROTOTYPES 

#include "CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectParameters.h"

using Parameters = Visualization::OpenGLInterface::OpenGLBufferObjectParameters;
using Visualization::OpenGLInterface::HandleGLError;

namespace Visualization
{
namespace OpenGLInterface
{

CreateOpenGLBuffer::CreateOpenGLBuffer(const Parameters& parameters):
  // https://stackoverflow.com/questions/14111900/using-new-on-void-pointer
  data_{::operator new(parameters.calculate_new_data_store_size())}
{}

CreateOpenGLBuffer::~CreateOpenGLBuffer()
{
  if (data_ != nullptr)
  {
    ::operator delete(data_);
  }
}

bool CreateOpenGLBuffer::create_buffer_object_data(
  const Parameters& parameters)
{
  glBufferData(
    parameters.binding_target_,
    parameters.calculate_new_data_store_size(),
    data_,
    parameters.usage_);

  HandleGLError gl_err {};
  return (gl_err() == "GL_NO_ERROR");  
}

} // namespace OpenGLInterface
} // namespace Visualization