#ifndef ALGEBRA_MODULES_OPTIMIZATION_CONJUGATE_GRADIENT_H
#define ALGEBRA_MODULES_OPTIMIZATION_CONJUGATE_GRADIENT_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"

#include <cstddef>
#include <optional>

namespace Algebra
{
namespace Modules
{
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

    std::optional<float> initial_guess(
      SparseMatrixMorphismOnDenseVector& morphism,
      CuBLASVectorOperations& vector_operations,
      CompressedSparseRowMatrix& A,
      DenseVector& x,
      DenseVector& Ax,
      Array& r_0);

  private:

    std::size_t maximum_iteration_;
    float tolerance_;
};

} // namespace Optimization
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_OPTIMIZATION_CONJUGATE_GRADIENT_H