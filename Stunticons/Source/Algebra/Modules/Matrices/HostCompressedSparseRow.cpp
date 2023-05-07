#include "HostCompressedSparseRow.h"

#include <cstddef> // std::size_t
#include <cstdlib> // free

namespace Algebra
{
namespace Modules
{
namespace Matrices
{

CStyleHostCompressedSparseRowMatrix::CStyleHostCompressedSparseRowMatrix(
  const std::size_t M,
  const std::size_t N,
  const std::size_t number_of_elements
  ):
  value_{static_cast<float*>(malloc(number_of_elements * sizeof(float)))},
  J_{static_cast<int*>(malloc(number_of_elements * sizeof(int)))},
  I_{static_cast<int*>(malloc((M + 1) * sizeof(int)))},
  M_{M},
  N_{N},
  number_of_elements_{number_of_elements}
{}

CStyleHostCompressedSparseRowMatrix::~CStyleHostCompressedSparseRowMatrix()
{
  free(value_);
  free(J_);
  free(I_);
}

HostCompressedSparseRowMatrix::HostCompressedSparseRowMatrix(
  const std::size_t M,
  const std::size_t N,
  const std::size_t number_of_elements
  ):
  values_{new float[number_of_elements]},
  J_{new int[number_of_elements]},
  I_{new int[M + 1]},
  M_{M},
  N_{N},
  number_of_elements_{number_of_elements}
{}

HostCompressedSparseRowMatrix::~HostCompressedSparseRowMatrix()
{
  delete [] values_;
  delete [] J_;
  delete [] I_;
}

} // namespace Matrices
} // namespace Modules
} // namespace Algebra
