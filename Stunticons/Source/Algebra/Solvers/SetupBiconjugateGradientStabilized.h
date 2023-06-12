#ifndef ALGEBRA_SOLVERS_SETUP_BICONJUGATE_GRADIENT_STABILIZED_H
#define ALGEBRA_SOLVERS_SETUP_BICONJUGATE_GRADIENT_STABILIZED_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"

#include <cstddef>
#include <vector>

namespace Algebra
{
namespace Solvers
{

//------------------------------------------------------------------------------
/// \brief Setup to use Biconjugate Gradient Stabilized method.
//------------------------------------------------------------------------------
class SetupBiconjugateGradientStabilized
{
  public:

    using Array = Algebra::Modules::Vectors::DoubleArray;

    using CompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::DoubleCompressedSparseRowMatrix;

    using VectorOperations =
      Algebra::Modules::Vectors::DoubleCuBLASVectorOperations;

    using DenseVector =
      Algebra::Modules::Matrices::SparseMatrices::DoubleDenseVector;

    using HostCompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::
        DoubleHostCompressedSparseRowMatrix;

    using SparseMatrixMorphismOnDenseVector =
      Algebra::Modules::Morphisms::DoubleSparseMatrixMorphismOnDenseVector;

    SetupBiconjugateGradientStabilized() = delete;

    SetupBiconjugateGradientStabilized(
      const HostCompressedSparseRowMatrix& host_csr,
      const std::vector<double>& host_b);

    ~SetupBiconjugateGradientStabilized() = default;

    std::size_t M_;
    CompressedSparseRowMatrix A_;
    DenseVector b_;
    DenseVector x_;
    DenseVector Ax_;
    SparseMatrixMorphismOnDenseVector morphism_;
    Array r_;
    VectorOperations operations_;
    DenseVector p_;
    DenseVector s_;
};

} // namespace Solvers
} // namespace Algebra

#endif // ALGEBRA_SOLVERS_SETUP_BICONJUGATE_GRADIENT_STABILIZED_H