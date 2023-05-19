#ifndef ALGEBRA_MODULES_MORPHISMS_CUBLAS_VECTOR_OPERATIONS_H
#define ALGEBRA_MODULES_MORPHISMS_CUBLAS_VECTOR_OPERATIONS_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Array.h"

#include "cublas_v2.h"
#include <cstddef> // std::size_t
#include <optional>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

class CuBLASVectorOperations
{
  public:

    using DenseVector = Algebra::Modules::Matrices::SparseMatrices::DenseVector;

    CuBLASVectorOperations();

    ~CuBLASVectorOperations();

    //--------------------------------------------------------------------------
    /// \details Multiples the vector x by scalar alpha and adds it to vector y
    /// overwriting the latest vector with the result.
    ///
    /// y[j] = \alpha * x[k] + y[j] for i = 1, \dots, n, with
    /// k = 1 + (i - 1) * incx, and
    /// j = 1 + (i - 1) * incy, where
    /// incx - stride between consecutive elements of x and
    /// incy - stride between consecutive elements of y.
    //--------------------------------------------------------------------------

    bool scalar_multiply_and_add_vector(
      const float alpha,
      const DenseVector& x,
      Array& y);

    //--------------------------------------------------------------------------
    /// \details y := \alpha * x + y
    //--------------------------------------------------------------------------
    bool scalar_multiply_and_add_vector(
      const float alpha,
      const DenseVector& x,
      DenseVector& y);

    bool scalar_multiply_and_add_vector(
      const float alpha,
      const Array& x,
      Array& y);

    bool scalar_multiply_and_add_vector(
      const float alpha,
      const Array& x,
      DenseVector& y);

    bool scalar_multiply(
      const float scalar,
      DenseVector& x,
      const std::size_t stride=1);

    bool scalar_multiply(
      const float scalar,
      Array& x,
      const std::size_t stride=1);

    // TODO: Figure out if the dot product, cublas<t>dot() are blocking
    // (synchronous) calls. It is expected that they should be.

    std::optional<float> dot_product(const Array& r1, const Array& r2);

    std::optional<float> dot_product(
      const DenseVector& r1,
      const DenseVector& r2);

    bool copy(const Array& x, DenseVector& y);

  private:

    cublasHandle_t cublas_handle_;
};

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_MORPHISMS_CUBLAS_VECTOR_OPERATIONS_H