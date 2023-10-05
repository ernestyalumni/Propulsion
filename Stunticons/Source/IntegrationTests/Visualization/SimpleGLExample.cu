#include "Utilities/HandleUnsuccessfulCudaCall.h"

// cudaGraphicsGLRegisterBuffer, cudaGraphicsMapFlagNone
#include <cuda_gl_interop.h> 
#include <cuda_runtime.h> // cudaFree, cudaMalloc, cudaMemcpyAsync
#include <GL/gl.h> // GLuint

int main()
{
  GLuint buffer_object {};
  cudaGraphicsResource* resource {nullptr};

  cudaGraphicsGLRegisterBuffer(
    &resource,
    buffer_object,
    cudaGraphicsMapFlagsNone);	
}