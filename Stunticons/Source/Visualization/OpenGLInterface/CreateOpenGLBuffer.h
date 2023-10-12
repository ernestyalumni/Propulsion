#ifndef VISUALIZATION_OPENGL_INTERFACE_CREATE_OPENGL_BUFFER_H
#define VISUALIZATION_OPENGL_INTERFACE_CREATE_OPENGL_BUFFER_H

#include "Visualization/OpenGLInterface/OpenGLBufferObjectParameters.h"

#include <GL/gl.h>
#include <cstddef>

namespace Visualization
{
namespace OpenGLInterface
{

class CreateOpenGLBuffer
{
  public:

    using Parameters =
      Visualization::OpenGLInterface::OpenGLBufferObjectParameters;

    //--------------------------------------------------------------------------
    /// \details If no data is to be copied, then the pointer to data that'll be
    /// copied into the data store for buffer data is null.
    //--------------------------------------------------------------------------
    CreateOpenGLBuffer():
      data_{nullptr}
    {}

    CreateOpenGLBuffer(const Parameters& parameters);

    ~CreateOpenGLBuffer();

    bool create_buffer_object_data(const Parameters& parameters);

  private:

    void* data_;
};

} // namespace OpenGLInterface
} // namespace Visualization

#endif // VISUALIZATION_OPEN_GL_INTERFACE_OPEN_GL_BUFFER_OBJECT_H
