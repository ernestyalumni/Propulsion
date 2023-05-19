#ifndef OPTIMIZATION_CONJUGATE_GRADIENT_H
#define OPTIMIZATION_CONJUGATE_GRADIENT_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"

#include <cstddef>
#include <optional>

namespace Optimization
{

class ConjugateGradient
{
  public:

    using Array = Algebra::Modules::Vectors::Array;

    using CompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;

    using CuBLASVectorOperations =
      Algebra::Modules::Vectors::CuBLASVectorOperations;

    using DenseVector =
      Algebra::Modules::Matrices::SparseMatrices::DenseVector;

    using SparseMatrixMorphismOnDenseVector =
      Algebra::Modules::Morphisms::SparseMatrixMorphismOnDenseVector;

    ConjugateGradient(
      const std::size_t maximum_iteration=10000,
      const float tolerance=1e-5f);

    ~ConjugateGradient() = default;

    // Initial guess for x_* such that Ax_* = b is x_* = 0. Otherwise, consider
    // Az = b - Ax and so since A is positive definite, z is guessed to be 0.
    static bool create_default_initial_guess(DenseVector& x);

    static std::optional<float> initial_step(
      SparseMatrixMorphismOnDenseVector& morphism,
      CuBLASVectorOperations& vector_operations,
      CompressedSparseRowMatrix& A,
      DenseVector& x,
      DenseVector& Ax,
      Array& r_0);

    //--------------------------------------------------------------------------
    /// \return If has_value(), returns r0, r1.
    //--------------------------------------------------------------------------

    static std::optional<std::tuple<float, float>> step(
      const std::size_t k,
      const float r0,
      const float r1,
      DenseVector& p,
      Array& r,
      SparseMatrixMorphismOnDenseVector& morphism,
      CuBLASVectorOperations& vector_operations,
      CompressedSparseRowMatrix& A,
      DenseVector& Ax,
      DenseVector& x);

    bool solve(
      DenseVector& p,
      Array& r,
      SparseMatrixMorphismOnDenseVector& morphism,
      CuBLASVectorOperations& vector_operations,
      CompressedSparseRowMatrix& A,
      DenseVector& Ax,
      DenseVector& x);

  private:

    std::size_t maximum_iteration_;
    float tolerance_;
};

} // namespace Optimization

#endif // OPTIMIZATION_CONJUGATE_GRADIENT_H