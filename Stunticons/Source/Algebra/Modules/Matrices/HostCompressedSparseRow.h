#ifndef ALGEBRA_MODULES_MATRICES_HOST_COMPRESSED_SPARSE_ROW_H
#define ALGEBRA_MODULES_MATRICES_HOST_COMPRESSED_SPARSE_ROW_H

#include <cstddef> // std::size_t

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