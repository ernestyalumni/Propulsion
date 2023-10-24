#include "Array.h"
#include "HostArrays.h"
#include "Utilities/HandleUnsuccessfulCudaCall.h"

#include <cstddef> // std::size_t
#include <cuda_runtime.h> // cudaFree, cudaMalloc, cudaMemcpyAsync
#include <stdexcept>
#include <string>
#include <vector>

using Utilities::HandleUnsuccessfulCUDACall;
using std::size_t;
using std::vector;

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

Array::Array(
  const std::size_t input_size
  ):
  values_{nullptr},
  number_of_elements_{input_size}
{
  const size_t size_in_bytes {input_size * sizeof(float)};

  HandleUnsuccessfulCUDACall handle_malloc {"Failed to allocate device array"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_malloc,
    cudaMalloc(reinterpret_cast<void**>(&values_), size_in_bytes));

  if (!handle_malloc.is_cuda_success())
  {
    throw std::runtime_error(std::string{handle_malloc.get_error_message()});
  }
}

Array::~Array()
{
  HandleUnsuccessfulCUDACall handle_free {"Failed to free device array"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_free,
    cudaFree(values_));
}

bool Array::copy_host_input_to_device(const HostArray& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from host to device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_values,
    cudaMemcpy(
      values_,
      h_a.values_,
      h_a.number_of_elements_ * sizeof(float),
      cudaMemcpyHostToDevice));

  return handle_values.is_cuda_success();
}

bool Array::copy_device_output_to_host(HostArray& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from device to host"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_values,
    cudaMemcpy(
      h_a.values_,
      values_,
      number_of_elements_ * sizeof(float),
      cudaMemcpyDeviceToHost));

  return handle_values.is_cuda_success();
}

bool Array::copy_host_input_to_device(const vector<float>& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from host to device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_values,
    cudaMemcpy(
      values_,
      h_a.data(),
      h_a.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  return handle_values.is_cuda_success();
}

bool Array::copy_device_output_to_host(vector<float>& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from device to host"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_values,
    cudaMemcpy(
      h_a.data(),
      values_,
      number_of_elements_ * sizeof(float),
      cudaMemcpyDeviceToHost));

  return handle_values.is_cuda_success();
}

bool Array::asynchronous_copy_host_input_to_device(
  const HostArray& h_a,
  cudaStream_t stream)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to asynchronously copy values from host to device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_values,
    cudaMemcpyAsync(
      values_,
      h_a.values_,
      h_a.number_of_elements_ * sizeof(float),
      cudaMemcpyDefault,
      stream));

  return handle_values.is_cuda_success();
}

bool Array::asynchronous_copy_device_output_to_host(
  HostArray& h_a,
  cudaStream_t stream)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to asynchronously copy values from device to host"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_values,
    cudaMemcpyAsync(
      h_a.values_,
      values_,
      number_of_elements_ * sizeof(float),
      cudaMemcpyDefault,
      stream));

  return handle_values.is_cuda_success();
}

DoubleArray::DoubleArray(
  const std::size_t input_size
  ):
  values_{nullptr},
  number_of_elements_{input_size}
{
  const size_t size_in_bytes {input_size * sizeof(double)};

  HandleUnsuccessfulCUDACall handle_malloc {"Failed to allocate device array"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_malloc,
    cudaMalloc(reinterpret_cast<void**>(&values_), size_in_bytes));

  if (!handle_malloc.is_cuda_success())
  {
    throw std::runtime_error(std::string{handle_malloc.get_error_message()});
  }
}

DoubleArray::~DoubleArray()
{
  HandleUnsuccessfulCUDACall handle_free {"Failed to free device array"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_free,
    cudaFree(values_));
}

bool DoubleArray::copy_host_input_to_device(const DoubleHostArray& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from host to device"};

  handle_values(cudaMemcpy(
    values_,
    h_a.values_,
    h_a.number_of_elements_ * sizeof(double),
    cudaMemcpyHostToDevice));

  return handle_values.is_cuda_success();
}

bool DoubleArray::copy_device_output_to_host(DoubleHostArray& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from device to host"};

  handle_values(cudaMemcpy(
    h_a.values_,
    values_,
    number_of_elements_ * sizeof(double),
    cudaMemcpyDeviceToHost));

  return handle_values.is_cuda_success();
}

bool DoubleArray::copy_host_input_to_device(const vector<double>& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from host to device"};

  handle_values(cudaMemcpy(
    values_,
    h_a.data(),
    h_a.size() * sizeof(double),
    cudaMemcpyHostToDevice));

  return handle_values.is_cuda_success();
}

bool DoubleArray::copy_device_output_to_host(vector<double>& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from device to host"};

  handle_values(cudaMemcpy(
    h_a.data(),
    values_,
    number_of_elements_ * sizeof(float),
    cudaMemcpyDeviceToHost));

  return handle_values.is_cuda_success();
}

bool DoubleArray::asynchronous_copy_host_input_to_device(
  const DoubleHostArray& h_a,
  cudaStream_t stream)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to asynchronously copy values from host to device"};

  handle_values(cudaMemcpyAsync(
    values_,
    h_a.values_,
    h_a.number_of_elements_ * sizeof(double),
    cudaMemcpyDefault,
    stream));

  return handle_values.is_cuda_success();
}

bool DoubleArray::asynchronous_copy_device_output_to_host(
  DoubleHostArray& h_a,
  cudaStream_t stream)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to asynchronously copy values from device to host"};

  handle_values(cudaMemcpyAsync(
    h_a.values_,
    values_,
    number_of_elements_ * sizeof(double),
    cudaMemcpyDefault,
    stream));

  return handle_values.is_cuda_success();
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra