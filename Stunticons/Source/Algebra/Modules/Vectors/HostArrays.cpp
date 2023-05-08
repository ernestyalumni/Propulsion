#include "HostArrays.h"

#include <cstddef> // std::size_t
#include <cstdlib> // free

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

CStyleHostArray::CStyleHostArray(
  const std::size_t input_size
  ):
  number_of_elements_{input_size},
  values_{static_cast<float*>(malloc(input_size * sizeof(float)))}
{}

CStyleHostArray::~CStyleHostArray()
{
  free(values_);
}

//------------------------------------------------------------------------------
/// \details When allocating memory with malloc, the ctor of the object is not
/// called. Likewise, when using free, dtor is not called.
/// When using new, ctor is called, and for delete, dtor is called.
/// Type Safety: malloc returns a (void*) ptr which has to be cast, but new
/// returns a pointer of correct type.
/// Memory Allocation Failure: If malloc fails, returns a null pointer.
/// If new fails, it throws std::bad_alloc.
/// Operator Overloading: new and delete can be overloaded; malloc, free can't.
//------------------------------------------------------------------------------

HostArray::HostArray(
  const std::size_t input_size
  ):
  values_{new float[input_size]},
  number_of_elements_{input_size}
{}

HostArray::~HostArray()
{
  delete [] values_;
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra