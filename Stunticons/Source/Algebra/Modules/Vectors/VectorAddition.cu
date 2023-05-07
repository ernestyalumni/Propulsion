#include "DeviceVectorAdditionArrays.h"
#include "HostVectorAdditionArrays.h"
#include "VectorAddition.h"

#include <cstddef> // std::size_t
#include <cstdint>
#include <iostream> // std::cerr
#include <cuda_runtime.h>

using std::cerr;
using std::size_t;

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

__global__ void vector_addition(
  const float* A,
  const float* B,
  float* C,
  size_t number_of_elements)
{
	const std::size_t i {blockDim.x * blockIdx.x + threadIdx.x};

  if (i < number_of_elements)
  {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

void vector_addition(
  HostVectorAdditionArrays& h_arrays,
  DeviceVectorAdditionArrays& d_arrays,
  const std::size_t threads_per_block)
{
  copy_host_input_to_device(h_arrays, d_arrays);

  const size_t blocks_per_grid {
    (h_arrays.number_of_elements_ + threads_per_block - 1) / threads_per_block};

  vector_addition<<<blocks_per_grid, threads_per_block>>>(
    d_arrays.d_A_,
    d_arrays.d_B_,
    d_arrays.d_C_,
    d_arrays.number_of_elements_);

  const cudaError_t err {cudaGetLastError()};

  if (err != cudaSuccess)
  {
    cerr << "Failed to launch vector_addition kernel (error code " <<
      cudaGetErrorString(err) << ")!\n";
  }

  copy_device_output_to_host(d_arrays, h_arrays);
}

void copy_host_input_to_device(
  const HostVectorAdditionArrays& hab,
  DeviceVectorAdditionArrays& dab)
{
  const cudaError_t err_A {
    cudaMemcpy(
      dab.d_A_,
      hab.h_A_,
      hab.number_of_elements_ * sizeof(float),
      cudaMemcpyHostToDevice)};

  const cudaError_t err_B {
    cudaMemcpy(
      dab.d_B_,
      hab.h_B_,
      hab.number_of_elements_ * sizeof(float),
      cudaMemcpyHostToDevice)};

  if (err_A != cudaSuccess)
  {
    cerr << "Failed to copy array A from host to device (error code " <<
      cudaGetErrorString(err_A) << ")!\n";
  }
  if (err_B != cudaSuccess)
  {
    cerr << "Failed to copy array B from host to device (error code " <<
      cudaGetErrorString(err_B) << ")!\n";
  }
}

void copy_device_output_to_host(
  const DeviceVectorAdditionArrays& dc,
  HostVectorAdditionArrays& hc)
{
  const cudaError_t err_C {
    cudaMemcpy(
      hc.h_C_,
      dc.d_C_,
      dc.number_of_elements_ * sizeof(float),
      cudaMemcpyDeviceToHost)};

  if (err_C != cudaSuccess)
  {
    cerr << "Failed to copy array C from device to host (error code " <<
      cudaGetErrorString(err_C) << ")!\n";
  }
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
