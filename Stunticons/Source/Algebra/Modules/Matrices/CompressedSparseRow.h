#ifndef ALGEBRA_MODULES_MATRICES_COMPRESSED_SPARSE_ROW_H
#define ALGEBRA_MODULES_MATRICES_COMPRESSED_SPARSE_ROW_H

#include "Algebra/Modules/Vectors/HostArrays.h"
#include "HostCompressedSparseRow.h"

#include <cstddef>
#include <cusparse.h> // cusparseSpMatDescr_t
#include <vector>

namespace Algebra
{
namespace Modules
{
namespace Matrices
{
namespace SparseMatrices
{

class CompressedSparseRowMatrix
{
  public:

    CompressedSparseRowMatrix() = delete;

    explicit CompressedSparseRowMatrix(
      const std::size_t M,
      const std::size_t N,
      const std::size_t number_of_elements);

    ~CompressedSparseRowMatrix();

    void copy_host_input_to_device(const HostCompressedSparseRowMatrix& h_a);

    void copy_device_output_to_host(HostCompressedSparseRowMatrix& h_a);

    float* d_values_;
    int* d_columns_;
    int* d_rows_;

    const std::size_t M_;
    const std::size_t N_;
    const std::size_t number_of_elements_;    

    cusparseSpMatDescr_t matrix_descriptor_;
};

class DenseVector
{
  public:

    DenseVector() = delete;

    explicit DenseVector(const std::size_t N);

    ~DenseVector();

    void copy_host_input_to_device(
      const Algebra::Modules::Vectors::HostArray& h_a);

    void copy_device_output_to_host(Algebra::Modules::Vectors::HostArray& h_a);

    void copy_host_input_to_device(const std::vector<float>& h_a);

    void copy_device_output_to_host(std::vector<float>& h_a);

    float* d_values_;

    const std::size_t number_of_elements_;

    cusparseDnVecDescr_t vector_descriptor_;
};

} // namespace SparseMatrices
} // namespace Matrices
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_MATRICES_COMPRESSED_SPARSE_ROW_H