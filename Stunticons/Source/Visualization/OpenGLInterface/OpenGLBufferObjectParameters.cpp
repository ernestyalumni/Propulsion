#include "OpenGLBufferObjectParameters.h"

#include <GL/gl.h>
#include <cstddef>

using std::size_t;

namespace Visualization
{
namespace OpenGLInterface
{

std::array<GLenum, 6> OpenGLBufferObjectParameters::valid_targets_ {
  GL_ARRAY_BUFFER,
  GL_ATOMIC_COUNTER_BUFFER,
  GL_PIXEL_PACK_BUFFER,
  GL_PIXEL_UNPACK_BUFFER,
  GL_TEXTURE_BUFFER,
  GL_PIXEL_UNPACK_BUFFER_ARB
};

std::array<GLenum, 4> OpenGLBufferObjectParameters::valid_usage_ {
  GL_STREAM_DRAW,
  GL_STREAM_READ,
  GL_DYNAMIC_DRAW,
  GL_DYNAMIC_DRAW_ARB
};

OpenGLBufferObjectParameters::OpenGLBufferObjectParameters(
  const size_t number_of_buffer_object_names,
  const GLenum binding_target,
  const GLenum usage,
  const size_t width,
  const size_t height
  ):
  number_of_buffer_object_names_{number_of_buffer_object_names},
  binding_target_{binding_target},
  usage_{usage},
  width_{width},
  height_{height}
{}

OpenGLBufferObjectParameters::OpenGLBufferObjectParameters():
  // Choice of width and height as default values was arbitrary.
  OpenGLBufferObjectParameters{1, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, 512, 512}
{}

} // namespace OpenGLInterface
} // namespace Visualization