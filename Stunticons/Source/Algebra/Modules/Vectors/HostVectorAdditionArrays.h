#ifndef ALGEBRA_MODULES_VECTORS_HOST_VECTOR_ADDITION_ARRAYS_H
#define ALGEBRA_MODULES_VECTORS_HOST_VECTOR_ADDITION_ARRAYS_H

#include <cstddef>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
/// \brief Host helper struct for element by element vector addition.
/// \href https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAdd
//------------------------------------------------------------------------------
struct HostVectorAdditionArrays
{
  const std::size_t number_of_elements_;

  float* h_A_;
  float* h_B_;
  float* h_C_;

  HostVectorAdditionArrays(const std::size_t input_size = 50000);

  ~HostVectorAdditionArrays();
};

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_HOST_VECTOR_ADDITION_ARRAYS_H