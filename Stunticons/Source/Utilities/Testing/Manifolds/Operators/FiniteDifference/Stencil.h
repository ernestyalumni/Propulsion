#ifndef UTILITIES_TESTING_STENCIL_H
#define UTILITIES_TESTING_STENCIL_H

#include "Utilities/HandleUnsuccessfulCUDACall.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h> // For __half
#include <iostream> // std::cerr
#include <stdexcept>

namespace Utilities
{
namespace Testing
{
namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

template <typename FPT, int NU>
struct Stencil
{
  using HandleUnsuccessfulCUDACall = Utilities::HandleUnsuccessfulCUDACall;

  FPT (*stencil_)[2];
  bool is_cuda_freed_;

  Stencil():
    stencil_{nullptr},
    is_cuda_freed_{false}
  {
    HandleUnsuccessfulCUDACall handle_malloc {
      "Failed to allocate device array"};

    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_malloc,
      cudaMalloc(reinterpret_cast<void**>(&stencil_), sizeof(FPT) * NU * 2));

    if (!handle_malloc.is_cuda_success())
    {
      throw std::runtime_error(std::string{handle_malloc.get_error_message()});
    }
  }

  ~Stencil()
  {
    free_resources();
  }
  
  bool copy_host_input_to_device(const FPT stencil[NU][2])
  {
    HandleUnsuccessfulCUDACall handle_copy {
      "Failed to copy values from host to device"};

    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_copy,
      cudaMemcpy(
        stencil_,
        stencil,
        sizeof(FPT) * NU * 2,
        cudaMemcpyHostToDevice));

    return handle_copy.is_cuda_success();
  }

  bool copy_device_output_to_host(FPT stencil[NU][2])
  {
    HandleUnsuccessfulCUDACall handle_copy {
      "Failed to copy values from device to host"};
      
    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_copy,
      cudaMemcpy(
        stencil,
        stencil_,
        sizeof(FPT) * NU * 2,
        cudaMemcpyDeviceToHost));

    return handle_copy.is_cuda_success();
  }

  bool free_resources()
  {
    if ((stencil_ != nullptr) && (is_cuda_freed_ == false))
    {
      HandleUnsuccessfulCUDACall handle_free {
        "Failed to free device array"};

      HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
        handle_free,
        cudaFree(stencil_));

      if (!handle_free.is_cuda_success())
      {
        std::cerr << handle_free.get_error_message() << "\n";
      }
      else
      {
        is_cuda_freed_ = true;
      }

      return handle_free.is_cuda_success();
    }

    return false;
  }
};

// Explicit template declarations.

extern template struct Stencil<float, 1>;
extern template struct Stencil<float, 2>;
extern template struct Stencil<float, 3>;
extern template struct Stencil<float, 4>;

extern template struct Stencil<double, 1>;
extern template struct Stencil<double, 2>;
extern template struct Stencil<double, 3>;
extern template struct Stencil<double, 4>;

extern template struct Stencil<__half, 1>;
extern template struct Stencil<__half, 2>;
extern template struct Stencil<__half, 3>;
extern template struct Stencil<__half, 4>;

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
} // namespace Testing
} // namespace Utilities

#endif // UTILITIES_TESTING_STENCIL_H
