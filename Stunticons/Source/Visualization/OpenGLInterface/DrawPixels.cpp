#include "Visualization/OpenGLInterface/DrawPixels.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"

//#include <GL/gl.h>
#include <array>
#include <cstddef>

using Parameters = Visualization::OpenGLInterface::DrawPixels::Parameters;
using Visualization::OpenGLInterface::HandleGLError;

namespace Visualization
{
namespace OpenGLInterface
{

std::array<GLenum, 13> DrawPixels::Parameters::valid_format_ {
  GL_COLOR_INDEX,
  GL_STENCIL_INDEX,
  GL_DEPTH_COMPONENT,
  GL_RGB,
  GL_BGR,
  GL_RGBA,
  GL_BGRA,
  GL_RED,
  GL_GREEN,
  GL_BLUE,
  GL_ALPHA,
  GL_LUMINANCE,
  GL_LUMINANCE_ALPHA
};

std::array<GLenum, 21> DrawPixels::Parameters::valid_type_ {
  GL_UNSIGNED_BYTE,
  GL_BYTE,
  GL_BITMAP,
  GL_UNSIGNED_SHORT,
  GL_SHORT,
  GL_UNSIGNED_INT,
  GL_INT,
  GL_FLOAT,
  GL_UNSIGNED_BYTE_3_3_2,
  GL_UNSIGNED_BYTE_2_3_3_REV,
  GL_UNSIGNED_SHORT_5_6_5,
  GL_UNSIGNED_SHORT_5_6_5_REV,
  GL_UNSIGNED_SHORT_4_4_4_4,
  GL_UNSIGNED_SHORT_4_4_4_4_REV,
  GL_UNSIGNED_SHORT_5_5_5_1,
  GL_UNSIGNED_SHORT_1_5_5_5_REV,
  GL_UNSIGNED_INT_8_8_8_8,
  GL_UNSIGNED_INT_8_8_8_8_REV,
  GL_UNSIGNED_INT_10_10_10_2,
  GL_UNSIGNED_INT_2_10_10_10_REV,
  // This enum value was not in
  // https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glDrawPixels.xml
  GL_UNSIGNED_INT_10_10_10_2_EXT,  
};

HandleGLError DrawPixels::draw_pixels_to_frame_buffer(
  const Parameters& parameters)
{
  HandleGLError gl_error {};

  glDrawPixels(
    parameters.width_,
    parameters.height_,
    parameters.format_,
    parameters.type_,
    nullptr);

  gl_error();

  return gl_error;
}

} // namespace OpenGLInterface
} // namespace Visualization
