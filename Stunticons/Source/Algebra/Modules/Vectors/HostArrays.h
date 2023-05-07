#ifndef ALGEBRA_MODULES_VECTORS_HOST_ARRAYS_H
#define ALGEBRA_MODULES_VECTORS_HOST_ARRAYS_H

#include <cstddef>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
/// \brief Host helper struct for arrays allocated on the host.
/// \href https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/vectorAdd
//------------------------------------------------------------------------------
struct CStyleHostArray
{
  const std::size_t number_of_elements_;

  float* values_;

  CStyleHostArray(const std::size_t input_size = 50000);

  ~CStyleHostArray();
};

struct HostArray
{
  float* values_;
  const std::size_t number_of_elements_;

  HostArray(const std::size_t input_size = 50000);

  ~HostArray();
};

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_HOST_VECTOR_ADDITION_ARRAYS_H