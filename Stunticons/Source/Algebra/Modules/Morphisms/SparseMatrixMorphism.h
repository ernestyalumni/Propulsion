#ifndef ALGEBRA_MODULES_MORPHISMS_SPARSE_MATRIX_MORPHISM_H
#define ALGEBRA_MODULES_MORPHISMS_SPARSE_MATRIX_MORPHISM_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"

#include <cstddef>
#include <cusparse.h> // cusparseSpMatDescr_t

namespace Algebra
{
namespace Modules
{
namespace Morphisms
{

class SparseMatrixMorphismOnDenseVectors
{
  public:

    using CompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;
    using DenseVector = Algebra::Modules::Matrices::SparseMatrices::DenseVector;

    SparseMatrixMorphismOnDenseVectors(
      const float alpha = 1.0,
      const float beta = 0.0);

    ~SparseMatrixMorphismOnDenseVectors();

    float get_alpha() const
    {
      return alpha_;
    }

    float get_beta() const
    {
      return beta_;
    }

    std::size_t get_buffer_size() const
    {
      return buffer_size_;
    }

    //--------------------------------------------------------------------------
    /// \details
    ///
    /// Y = \alpha op(A) \cdot X + \beta Y
    ///
    /// where \alpha, \beta are scalars,
    /// op(A) is a sparse matrix of size m x k,
    /// X, Y are dense vectors of sizes k, m, respectively,
    /// op(A) == A if op(A) == CUSPARSE_OPERATION_NON_TRANSPOSE
    /// A^T if op(A) == CUSPARSE_OPERATION_TRANSPOSE
    /// A^H if op(A) == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
    /// \ref https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv
    /// \returns True if successful, false, if not.
    //--------------------------------------------------------------------------  
    bool linear_transform(
      CompressedSparseRowMatrix& A,
      DenseVector& x,
      DenseVector& b);

    //--------------------------------------------------------------------------
    /// \returns True if successful, false, if not.
    //--------------------------------------------------------------------------
    bool buffer_size(
      CompressedSparseRowMatrix& A,
      DenseVector& x,
      DenseVector& b);

  private:

    void* buffer_;
    cusparseHandle_t cusparse_handle_;
    std::size_t buffer_size_;

    float alpha_;
    float beta_;
};

} // namespace Morphisms
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_MORPHISMS_SPARSE_MATRIX_MORPHISM_H