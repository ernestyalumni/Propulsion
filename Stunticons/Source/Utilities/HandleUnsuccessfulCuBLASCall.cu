#include "HandleUnsuccessfulCuBLASCall.h"

#include "cublas_v2.h"
#include <iostream> // std::cerr
#include <string>

using std::cerr;

namespace Utilities
{

HandleUnsuccessfulCuBLASCall::HandleUnsuccessfulCuBLASCall(
  const std::string& error_message
  ):
  error_message_{error_message},
  cuBLAS_status_{CUBLAS_STATUS_SUCCESS}
{}

void HandleUnsuccessfulCuBLASCall::operator()(
  const cublasStatus_t cublas_status)
{
  cuBLAS_status_ = cublas_status;

  if (!is_cuBLAS_success())
  {
    cerr << error_message_ << " (error code " <<
      // https://docs.nvidia.com/cuda/cublas/index.html#cublasgetstatusstring
      cublasGetStatusString(cuBLAS_status_) << ")!\n";
  }
}

} // namespace Utilities