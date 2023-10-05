#include "OpenGLBufferObjectNames.h"

#include <GL/gl.h> // GLuint
#include <cstddef>

using std::size_t;

namespace Visualization
{
namespace OpenGLInterface
{

OpenGLBufferObjectNames::Parameters::Parameters(
  const std::size_t number_of_buffer_object_names,
  const GLenum binding_target
  ):
  number_of_buffer_object_names_{number_of_buffer_object_names},
  binding_target_{binding_target}
{}

OpenGLBufferObjectNames::Parameters::Parameters():
  Parameters{1, GL_ARRAY_BUFFER}
{}

OpenGLBufferObjectNames::OpenGLBufferObjectNames(const Parameters& parameters):
  parameters_{parameters},
  buffer_object_{}
{
  if (parameters.number_of_buffer_object_names_ > 1)
  {
    buffer_object_ = new GLuint[parameters.number_of_buffer_object_names_];
  }
}

OpenGLBufferObjectNames::~OpenGLBufferObjectNames()
{
  glDeleteBuffers(parameters_.number_of_buffer_object_names_, buffer_object_);

  if ((parameters_.number_of_buffer_object_names_ > 1) &&
    buffer_object_ != nullptr)
  {
    delete[] buffer_object_;
  }
}

void OpenGLBufferObjectNames::initialize()
{
  // Create buffer object.
  glGenBuffers(
    static_cast<GLsizei>(parameters_.number_of_buffer_object_names_),
    buffer_object_);

  if (parameters_.number_of_buffer_object_names_ > 1)
  {
    for (size_t i {0}; i < parameters_.number_of_buffer_object_names_; ++i)
    {
      glBindBuffer(parameters_.binding_target_, buffer_object_[i]);
    }
  }
  else
  {
    glBindBuffer(parameters_.binding_target_, *buffer_object_);
  }
}

} // namespace OpenGLInterface
} // namespace Visualization
