#ifndef ALGEBRA_MODULES_VECTORS_VECTOR_ADDITION_H
#define ALGEBRA_MODULES_VECTORS_VECTOR_ADDITION_H

#include "DeviceVectorAdditionArrays.h"
#include "HostVectorAdditionArrays.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
/// \brief Element by element vector addition.
/// \href https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAdd
//------------------------------------------------------------------------------
__global__ void vector_addition(
  const float* A,
  const float* B,
  float* C,
  std::size_t number_of_elements);

void copy_host_input_to_device(
  const HostVectorAdditionArrays& hab,
  DeviceVectorAdditionArrays& dab);

void copy_device_output_to_host(
  const DeviceVectorAdditionArrays& dc,
  HostVectorAdditionArrays& hc);

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_VECTOR_ADDITION_H