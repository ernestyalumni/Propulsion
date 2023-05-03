#ifndef ALGEBRA_MODULES_VECTORS_DEVICE_VECTOR_ADDITION_ARRAYS_H
#define ALGEBRA_MODULES_VECTORS_DEVICE_VECTOR_ADDITION_ARRAYS_H

#include <cstddef>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
/// \href https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAdd
//------------------------------------------------------------------------------
struct DeviceVectorAdditionArrays
{
  std::size_t number_of_elements_;
  float* d_A_;
  float* d_B_;
  float* d_C_;

  DeviceVectorAdditionArrays(const std::size_t input_size = 50000);

  ~DeviceVectorAdditionArrays();
};

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_DEVICE_VECTOR_ADDITION_ARRAYS_H