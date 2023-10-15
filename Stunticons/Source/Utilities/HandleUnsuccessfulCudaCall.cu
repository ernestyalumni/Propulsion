#include "HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h>
#include <iostream> // std::cerr
#include <string>

using std::cerr;

namespace Utilities
{

/*
HandleUnsuccessfulCUDACall::HandleUnsuccessfulCUDACall(
  const std::string& error_message
  ):
  error_message_{error_message},
  cuda_error_{cudaSuccess}
{}
*/

HandleUnsuccessfulCUDACall::HandleUnsuccessfulCUDACall(
  const std::string_view error_message
  ):
  error_message_{error_message},
  cuda_error_{cudaSuccess}
{}

HandleUnsuccessfulCUDACall::HandleUnsuccessfulCUDACall():
  error_message_{default_error_message_}
{}

void HandleUnsuccessfulCUDACall::operator()(const cudaError_t cuda_error)
{
  cuda_error_ = cuda_error;

  if (!is_cuda_success())
  {
    cerr << error_message_ << " (error code " <<
      cudaGetErrorString(cuda_error_) << ")!\n";
  }
}

void HandleUnsuccessfulCUDACall::operator()(
  const cudaError_t cuda_error,
  const char* filename,
  int line)
{
  cuda_error_ = cuda_error;

  if (!is_cuda_success())
  {
    std::string error_message {"File: "};
    error_message.append(filename);
    error_message.append(", line: ");
    error_message.append(std::to_string(line));
    error_message.append(", ");
    error_message.append(error_message_);
    error_message.append(" (error code ");
    error_message.append(cudaGetErrorString(cuda_error_));
    error_message.append(")!\n");

    error_message_ = error_message;
    cerr << error_message;
  }
}

} // namespace Utilities