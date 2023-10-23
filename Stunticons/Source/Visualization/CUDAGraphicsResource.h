#ifndef VISUALIZATION_CUDA_GRAPHICS_RESOURCE_H
#define VISUALIZATION_CUDA_GRAPHICS_RESOURCE_H

#include "Utilities/HandleUnsuccessfulCudaCall.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"

#include <array>

namespace Visualization
{

class CUDAGraphicsResource
{
  public:

    using HandleUnsuccessfulCUDACall = Utilities::HandleUnsuccessfulCUDACall;

    struct Parameters
    {
      //------------------------------------------------------------------------
      /// \ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g0fd33bea77ca7b1e69d1619caf44214b
      /// Used with cudaGraphicsGLRegisterBuffer, cudaGraphicsGLRegisterImage.
      //------------------------------------------------------------------------
      static std::array<unsigned int, 5> valid_flags_;

      Parameters(
        const unsigned int flags,
        const int count=1,
        cudaStream_t stream=0);

      Parameters():
        Parameters{cudaGraphicsMapFlagsNone}
      {}

      unsigned int flags_;

      int count_;
      cudaStream_t stream_;
    };

    CUDAGraphicsResource();
    CUDAGraphicsResource(const Parameters& parameters);

    //--------------------------------------------------------------------------
    /// \ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gc65d1f2900086747de1e57301d709940
    //--------------------------------------------------------------------------
    ~CUDAGraphicsResource();

    //------------------------------------------------------------------------
    /// \ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g0fd33bea77ca7b1e69d1619caf44214b
    /// __host__ cudaError_t cudaGraphicsGLRegisterBuffer(
    ///  cudaGraphicsResource** resource, GLuint buffer, unsigned int flags)
    //------------------------------------------------------------------------
    HandleUnsuccessfulCUDACall register_buffer_object(
      const Parameters& parameters,
      Visualization::OpenGLInterface::BufferObjectNames& name);

    HandleUnsuccessfulCUDACall unregister_buffer_object();

    bool is_registered() const
    {
      return is_registered_;
    }

    bool is_resource_mapped() const
    {
      return is_resource_mapped_;
    }

    //------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
    /// __host__ cudaError_t cudaGraphicsMapResources(int count,
    /// cudaGraphicsResource_t* resources, cudaStream_t stream=0)
    /// Map graphics resources for access by CUDA
    /// count - Number of resources to map
    /// resources - resources to map for CUDA
    /// stream - Stream for synchronization.
    /// This function provides synchronization guarantee that any graphics
    /// calls issued before cudaGraphicsMapResources() will complete before
    /// any subsequent CUDA work issued in stream begins.
    //------------------------------------------------------------------------
    Utilities::HandleUnsuccessfulCUDACall map_resource();

    //------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g11988ab4431b11ddb7cbde7aedb60491
    /// This function provides synchronization guarantee that any CUDA work
    /// issued in stream before cudaGraphicsUnmapResources() will complete
    /// before any subsequently issued graphics work begins.
    /// In other words, this call is important, prior to performing rendering,
    /// because it provides synchronization between CUDA and graphics portions
    /// of the application and cudaGraphicsUnmapResources() will complete
    /// before ensuing graphics calls begin.
    //------------------------------------------------------------------------
    Utilities::HandleUnsuccessfulCUDACall unmap_resource();

    friend HandleUnsuccessfulCUDACall get_mapped_device_pointer(
      CUDAGraphicsResource& cuda_graphics_resource);

    cudaGraphicsResource* cuda_graphics_resource_;

  private:

    Parameters parameters_;

    bool is_registered_;
    bool is_resource_mapped_;
};

} // namespace Visualization

#endif // VISUALIZATION_CUDA_GRAPHICS_RESOURCE_H