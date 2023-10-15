// needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers
#define GL_GLEXT_PROTOTYPES 

#include "OpenGLBufferObjectNames.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectParameters.h"

#include <GL/gl.h> // GLuint
#include <cstddef>

using Parameters = Visualization::OpenGLInterface::OpenGLBufferObjectParameters;
using Visualization::OpenGLInterface::HandleGLError;
using std::size_t;

namespace Visualization
{
namespace OpenGLInterface
{

OpenGLBufferObjectNames::OpenGLBufferObjectNames(const Parameters& parameters):
  parameters_{parameters},
  buffer_object_{},
  buffer_objects_{nullptr}
{
  if (parameters.number_of_buffer_object_names_ > 1)
  {
    buffer_objects_ = new GLuint[parameters.number_of_buffer_object_names_];
  }
}

OpenGLBufferObjectNames::~OpenGLBufferObjectNames()
{
  if ((parameters_.number_of_buffer_object_names_ > 1) &&
    buffer_objects_ != nullptr)
  {
    glDeleteBuffers(
      parameters_.number_of_buffer_object_names_,
      buffer_objects_);

    delete[] buffer_objects_;
  }
  else
  {
    glDeleteBuffers(
      parameters_.number_of_buffer_object_names_,
      &buffer_object_);
  }
}

bool OpenGLBufferObjectNames::initialize()
{
  if ((parameters_.number_of_buffer_object_names_ > 1) &&
    buffer_objects_ != nullptr)
  {
    // Create buffer object.
    glGenBuffers(
      static_cast<GLsizei>(parameters_.number_of_buffer_object_names_),
      buffer_objects_);

    for (size_t i {0}; i < parameters_.number_of_buffer_object_names_; ++i)
    {
      glBindBuffer(parameters_.binding_target_, buffer_objects_[i]);
    }
  }
  else
  {
    // Create buffer object.
    glGenBuffers(
      static_cast<GLsizei>(parameters_.number_of_buffer_object_names_),
      &buffer_object_);

    glBindBuffer(parameters_.binding_target_, buffer_object_);
  }

  HandleGLError gl_err {};
  return (gl_err() == "GL_NO_ERROR");
}

bool OpenGLBufferObjectNames::unbind_and_restore(const Parameters& parameters)
{
  HandleGLError gl_err {};

  glBindBuffer(parameters.binding_target_, 0);

  return (gl_err() == "GL_NO_ERROR");
}

} // namespace OpenGLInterface
} // namespace Visualization
