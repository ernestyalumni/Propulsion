#ifndef ALGEBRA_MODULES_MATRICES_COMPRESSED_SPARSE_ROW_H
#define ALGEBRA_MODULES_MATRICES_COMPRESSED_SPARSE_ROW_H

#include "Algebra/Modules/Vectors/HostArrays.h"
#include "HostCompressedSparseRow.h"

#include <array>
#include <cstddef>
#include <cuda_runtime.h> // cudaFree, cudaMalloc
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

//------------------------------------------------------------------------------
/// \brief Compressed Sparse Row (CSR) matrix for CUDA C++, but using double
/// floating points.
//------------------------------------------------------------------------------
class DoubleCompressedSparseRowMatrix
{
  public:

    DoubleCompressedSparseRowMatrix() = delete;

    explicit DoubleCompressedSparseRowMatrix(
      const std::size_t M,
      const std::size_t N,
      const std::size_t number_of_elements);

    ~DoubleCompressedSparseRowMatrix();

    void copy_host_input_to_device(
      const DoubleHostCompressedSparseRowMatrix& h_a);

    void copy_device_output_to_host(DoubleHostCompressedSparseRowMatrix& h_a);

    double* d_values_;
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

    bool copy_host_input_to_device(
      const Algebra::Modules::Vectors::HostArray& h_a);

    bool copy_device_output_to_host(Algebra::Modules::Vectors::HostArray& h_a);

    bool copy_host_input_to_device(const std::vector<float>& h_a);

    template <std::size_t N>
    bool copy_host_input_to_device(const std::array<float, N>& h_a)
    {
      const auto result = cudaMemcpy(
        d_values_,
        h_a.data(),
        h_a.size() * sizeof(float),
        cudaMemcpyHostToDevice);

      return result == cudaSuccess;
    }

    bool copy_device_output_to_host(std::vector<float>& h_a);

    template <std::size_t N>
    bool copy_device_output_to_host(std::array<float, N>& h_a)
    {
      const auto result = cudaMemcpy(
        d_values_,
        h_a.data(),
        h_a.size() * sizeof(float),
        cudaMemcpyHostToDevice);

      return result == cudaSuccess;
    }

    float* d_values_;

    const std::size_t number_of_elements_;

    cusparseDnVecDescr_t vector_descriptor_;
};

class DoubleDenseVector
{
  public:

    DoubleDenseVector() = delete;

    explicit DoubleDenseVector(const std::size_t N);

    ~DoubleDenseVector();

    bool copy_host_input_to_device(
      const Algebra::Modules::Vectors::DoubleHostArray& h_a);

    bool copy_device_output_to_host(
      Algebra::Modules::Vectors::DoubleHostArray& h_a);

    bool copy_host_input_to_device(const std::vector<double>& h_a);

    template <std::size_t N>
    bool copy_host_input_to_device(const std::array<double, N>& h_a)
    {
      const auto result = cudaMemcpy(
        d_values_,
        h_a.data(),
        h_a.size() * sizeof(float),
        cudaMemcpyHostToDevice);

      return result == cudaSuccess;
    }

    bool copy_device_output_to_host(std::vector<double>& h_a);

    template <std::size_t N>
    bool copy_device_output_to_host(std::array<double, N>& h_a)
    {
      const auto result = cudaMemcpy(
        d_values_,
        h_a.data(),
        h_a.size() * sizeof(float),
        cudaMemcpyHostToDevice);

      return result == cudaSuccess;
    }

    double* d_values_;

    const std::size_t number_of_elements_;

    cusparseDnVecDescr_t vector_descriptor_;
};

} // namespace SparseMatrices
} // namespace Matrices
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_MATRICES_COMPRESSED_SPARSE_ROW_H