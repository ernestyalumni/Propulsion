#include "CUDAGraphicsResource.h"
#include "Utilities/HandleUnsuccessfulCudaCall.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectNames.h"

#include <array>
// cudaGraphicsGLRegisterBuffer, cudaGraphicsMapFlagNone
#include <cuda_gl_interop.h> 

using Utilities::HandleUnsuccessfulCUDACall;
using Visualization::OpenGLInterface::OpenGLBufferObjectNames;

namespace Visualization
{

std::array<unsigned int, 5> CUDAGraphicsResource::Parameters::valid_flags_ {
  cudaGraphicsRegisterFlagsNone,
  cudaGraphicsRegisterFlagsReadOnly,
  cudaGraphicsRegisterFlagsWriteDiscard,
  cudaGraphicsRegisterFlagsSurfaceLoadStore,
  cudaGraphicsRegisterFlagsTextureGather
};

CUDAGraphicsResource::Parameters::Parameters(
  const unsigned int flags,
  const int count,
  cudaStream_t stream
  ):
  flags_{flags},
  count_{count},
  stream_{stream}
{}

CUDAGraphicsResource::CUDAGraphicsResource(const Parameters& parameters):
  cuda_graphics_resource_{},
  parameters_{parameters},
  is_registered_{false},
  is_resource_mapped_{false}
{}

CUDAGraphicsResource::CUDAGraphicsResource():
  CUDAGraphicsResource{Parameters{}}
{}

CUDAGraphicsResource::~CUDAGraphicsResource()
{
  if (is_registered_)
  {
    HandleUnsuccessfulCUDACall handle_unregister {
      "Failed to unregister CUDA resource"};

    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_unregister,
      cudaGraphicsUnregisterResource(
        cuda_graphics_resource_));
  }

  if (is_resource_mapped_)
  {
    unmap_resource();
  } 
}

HandleUnsuccessfulCUDACall
  CUDAGraphicsResource::register_buffer_object(
    const Parameters& parameters,
    OpenGLBufferObjectNames& name)
{
  HandleUnsuccessfulCUDACall handle_register {
    "Failed to register OpenGL buffer object"};

  is_registered_ = false;

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_register,
    cudaGraphicsGLRegisterBuffer(
      &cuda_graphics_resource_,
      name.buffer_object_,
      parameters.flags_));

  if (handle_register.is_cuda_success())
  {
    is_registered_ = true;
  }

  return handle_register;
}

HandleUnsuccessfulCUDACall CUDAGraphicsResource::map_resource()
{
  HandleUnsuccessfulCUDACall handle_map_resource {
    "Failed to map resource for access by CUDA"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_map_resource,
    cudaGraphicsMapResources(
      parameters_.count_,
      &cuda_graphics_resource_,
      parameters_.stream_));

  if (handle_map_resource.is_cuda_success())
  {
    is_resource_mapped_ = true;
  }

  return handle_map_resource;
}

HandleUnsuccessfulCUDACall CUDAGraphicsResource::unmap_resource()
{
  HandleUnsuccessfulCUDACall handle_unmap_resource {
    "Failed to unmap resource for access by CUDA"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_unmap_resource,
    cudaGraphicsUnmapResources(
      parameters_.count_,
      &cuda_graphics_resource_,
      parameters_.stream_));

  if (handle_unmap_resource.is_cuda_success())
  {
    is_resource_mapped_ = false;
  }

  return handle_unmap_resource;
}

} // namespace Visualization