#ifndef VISUALIZATION_MAPPED_DEVICE_POINTER_H
#define VISUALIZATION_MAPPED_DEVICE_POINTER_H

#include "Utilities/HandleUnsuccessfulCUDACall.h"
#include "Visualization/CUDAGraphicsResource.h"

#include <cstddef> // std::size_t
#include <cuda_gl_interop.h>

namespace Visualization
{

template <typename T>
class MappedDevicePointer
{
  public:

    using HandleUnsuccessfulCUDACall = Utilities::HandleUnsuccessfulCUDACall;

    MappedDevicePointer():
      device_pointer_{},
      size_{}
    {}

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1ga36881081c8deb4df25c256158e1ac99
    /// __host__ cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr,
    /// size_t* size, cudaGraphicsResource_t resource)
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCUDACall get_mapped_device_pointer(
      CUDAGraphicsResource& cuda_graphics_resource)
    {
      HandleUnsuccessfulCUDACall handle_get_mapped_pointer {
        "Failed to get device pointer to which to access graphics resource"};

      HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
        handle_get_mapped_pointer,
        cudaGraphicsResourceGetMappedPointer(
          reinterpret_cast<void**>(&device_pointer_),
          &size_,
          cuda_graphics_resource.cuda_graphics_resource_));

      return handle_get_mapped_pointer;
    }

    inline std::size_t get_size() const
    {
      return size_;
    }

    T* device_pointer_;

  private:

    std::size_t size_;
};

} // namespace Visualization

#endif // VISUALIZATION_MAPPED_DEVICE_POINTER_H