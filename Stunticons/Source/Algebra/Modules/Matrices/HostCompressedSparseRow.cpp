#include "HostCompressedSparseRow.h"

#include <algorithm>
#include <cstddef> // std::size_t
#include <cstdlib> // free
#include <vector>

using std::copy;
using std::vector;

namespace Algebra
{
namespace Modules
{
namespace Matrices
{
namespace SparseMatrices
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

auto HostCompressedSparseRowMatrix::copy_values(
  const vector<float>& input_values)
{
  return copy(input_values.begin(), input_values.end(), values_);
}

const int* HostCompressedSparseRowMatrix::copy_row_offsets(
  const vector<int>& row_offsets)
{
  return copy(row_offsets.begin(), row_offsets.end(), I_);
}

auto HostCompressedSparseRowMatrix::copy_column_indices(
  const vector<int>& column_indices)
{
  return copy(column_indices.begin(), column_indices.end(), J_);
}

} // namespace SparseMatrices
} // namespace Matrices
} // namespace Modules
} // namespace Algebra
