#include "GenerateCUDAImage.h"

#include "simpleCUDA2GL.h"
#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/MappedDevicePointer.h"

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

void GenerateCUDAImage::generate_CUDA_image(
  ::Visualization::CUDAGraphicsResource& cuda_graphics_resource,
  const dim3 threads,
  const dim3 blocks,
  const int image_width)
{
  mapped_device_pointer_.get_mapped_device_pointer(cuda_graphics_resource);

  make_striped_pattern(
    threads,
    blocks,
    mapped_device_pointer_.device_pointer_,
    image_width);
}

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests