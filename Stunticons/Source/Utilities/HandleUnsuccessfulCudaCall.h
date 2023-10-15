#ifndef UTILITIES_HANDLE_UNSUCCESSFUL_CUDA_CALL_H
#define UTILITIES_HANDLE_UNSUCCESSFUL_CUDA_CALL_H

#include <cuda_runtime.h>
#include <string>
#include <string_view>

namespace Utilities
{

#define HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(object, cuda_error) \
  object.operator()(cuda_error, __FILE__, __LINE__)

class HandleUnsuccessfulCUDACall
{
  public:

    inline static const std::string_view default_error_message_ {
      "cudaSuccess was not returned."};

    HandleUnsuccessfulCUDACall();

    HandleUnsuccessfulCUDACall(const std::string_view error_message);

    ~HandleUnsuccessfulCUDACall() = default;

    inline bool is_cuda_success() const
    {
      return cuda_error_ == cudaSuccess;
    }

    void operator()(const cudaError_t cuda_error);

    void operator()(
      const cudaError_t cuda_error,
      const char* filename,
      int line);

    cudaError_t get_cuda_error() const
    {
      return cuda_error_;
    }

  private:

    std::string_view error_message_;

    // ref. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6
    // enum cudaError - CUDA error types.
    // cudaSuccess = 0
    // cudaErrorInvalidValue = 1
    // This indicates 1 or more of parameters passed to API call isn't within an
    // acceptable range of values.
    cudaError_t cuda_error_;
};

} // namespace Utilities

#endif // UTILITIES_HANDLE_UNSUCCESSFUL_CUDA_CALL_H
