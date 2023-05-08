#ifndef UTILITIES_HANDLE_UNSUCCESSFUL_CUBLAS_CALL_H
#define UTILITIES_HANDLE_UNSUCCESSFUL_CUBLAS_CALL_H

#include "cublas_v2.h"
#include <string>

namespace Utilities
{

class HandleUnsuccessfulCuBLASCall
{
  public:

    inline static const std::string default_error_message_ {
      "cuBLAS Success was not returned."};

    HandleUnsuccessfulCuBLASCall(
      const std::string& error_message = default_error_message_);

    ~HandleUnsuccessfulCuBLASCall() = default;

    inline bool is_cuBLAS_success()
    {
      return cuBLAS_status_ == CUBLAS_STATUS_SUCCESS;
    }

    void operator()(const cublasStatus_t cuBLAS_status);

    cublasStatus_t get_cuBLAS_status() const
    {
      return cuBLAS_status_;
    }

  private:

    std::string error_message_;

    cublasStatus_t cuBLAS_status_;
};

} // namespace Utilities

#endif // UTILITIES_HANDLE_UNSUCCESSFUL_CUDA_CALL_H
