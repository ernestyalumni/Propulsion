#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_GENERATE_CUDA_IMAGE_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_GENERATE_CUDA_IMAGE_H

#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/MappedDevicePointer.h"

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

class GenerateCUDAImage
{
  public:

    GenerateCUDAImage():
      mapped_device_pointer_{}
    {}

    void generate_CUDA_image(
      ::Visualization::CUDAGraphicsResource& cuda_graphics_resource,
      const dim3 threads,
      const dim3 blocks,
      const int image_width);

  private:

    ::Visualization::MappedDevicePointer<unsigned int> mapped_device_pointer_;
};

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_GENERATE_CUDA_IMAGE_H