#ifndef ALGEBRA_MODULES_MATRICES_HOST_COMPRESSED_SPARSE_ROW_H
#define ALGEBRA_MODULES_MATRICES_HOST_COMPRESSED_SPARSE_ROW_H

#include <algorithm>
#include <array>
#include <cstddef> // std::size_t
#include <vector>

namespace Algebra
{
namespace Modules
{
namespace Matrices
{
namespace SparseMatrices
{

class CStyleHostCompressedSparseRowMatrix
{
  public:

    CStyleHostCompressedSparseRowMatrix() = delete;

    explicit CStyleHostCompressedSparseRowMatrix(
      const std::size_t M,
      const std::size_t N,
      const std::size_t number_of_elements);

    ~CStyleHostCompressedSparseRowMatrix();

    float* value_;
    int* J_;
    int* I_;
    const std::size_t M_;
    const std::size_t N_;
    const std::size_t number_of_elements_;
};

class HostCompressedSparseRowMatrix
{
  public:

    HostCompressedSparseRowMatrix() = delete;

    explicit HostCompressedSparseRowMatrix(
      const std::size_t M,
      const std::size_t N,
      const std::size_t number_of_elements);

    ~HostCompressedSparseRowMatrix();

    auto copy_values(const std::vector<float>& input_values);

    template <std::size_t NNZ>
    auto copy_values(const std::array<float, NNZ>& input_values)
    {
      return std::copy(input_values.begin(), input_values.end(), values_);
    }

    const int* copy_row_offsets(const std::vector<int>& row_offsets);

    template <std::size_t Mp1>
    auto copy_row_offsets(const std::array<int, Mp1>& row_offsets)
    {
      return std::copy(row_offsets.begin(), row_offsets.end(), I_);
    }

    auto copy_column_indices(const std::vector<int>& column_indices);

    template <std::size_t NNZ>
    auto copy_column_indices(const std::array<int, NNZ>& column_indices)
    {
      return std::copy(column_indices.begin(), column_indices.end(), J_);
    }

    float* values_;
    int* J_;
    int* I_;
    const std::size_t M_;
    const std::size_t N_;
    const std::size_t number_of_elements_;
};

} // namespace SparseMatrices
} // namespace Matrices
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_MATRICES_HOST_COMPRESSED_SPARSE_ROW_H