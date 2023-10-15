#ifndef VISUALIZATION_OPENGL_INTERFACE_OPENGL_BUFFER_OBJECT_NAMES_H
#define VISUALIZATION_OPENGL_INTERFACE_OPENGL_BUFFER_OBJECT_NAMES_H

#include "Visualization/OpenGLInterface/OpenGLBufferObjectParameters.h"

#include <GL/gl.h> // GLuint
#include <cstddef> // std::size_t

namespace Visualization
{
namespace OpenGLInterface
{

struct OpenGLBufferObjectNames
{
  using Parameters =
    Visualization::OpenGLInterface::OpenGLBufferObjectParameters;

  OpenGLBufferObjectNames(const Parameters& parameters);

  virtual ~OpenGLBufferObjectNames();

  //----------------------------------------------------------------------------
  /// \return True if no glError, return false if there was a glError.
  //----------------------------------------------------------------------------
  bool initialize();

  //----------------------------------------------------------------------------
  /// \details Unbind buffer object and restore client memory usage for the
  /// binding target data member in the parameters.
  //----------------------------------------------------------------------------
  static bool unbind_and_restore(const Parameters& parameters);

  //----------------------------------------------------------------------------
  /// \ref https://docs.gl/gl4/glBindBuffer
  /// void glBindBuffer(GLenum target, GLuint buffer);
  /// buffer set to zero effectively unbinds any buffer object previously bound
  /// and restores client memory usage for that buffer object target.
  //----------------------------------------------------------------------------
  inline bool unbind_and_restore()
  {
    return unbind_and_restore(parameters_);
  }

  const Parameters parameters_;

  // Exclusively for use as a single buffer object name.
  GLuint buffer_object_;

  // Exclusively for use as multiple (more than 1) buffer object names.
  GLuint* buffer_objects_;
};

} // namespace OpenGLInterface
} // namespace Visualization

#endif // VISUALIZATION_OPENGL_INTERFACE_OPENGL_BUFFER_OBJECT_NAMES_H
