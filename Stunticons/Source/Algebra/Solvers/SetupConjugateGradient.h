#ifndef ALGEBRA_SOLVERS_SETUP_CONJUGATE_GRADIENT_H
#define ALGEBRA_SOLVERS_SETUP_CONJUGATE_GRADIENT_H

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
class SetupConjugateGradient
{
  public:

    using Array = Algebra::Modules::Vectors::Array;

    using CompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;

    using VectorOperations = Algebra::Modules::Vectors::CuBLASVectorOperations;

    using DenseVector = Algebra::Modules::Matrices::SparseMatrices::DenseVector;

    using HostCompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix;

    using SparseMatrixMorphismOnDenseVector =
      Algebra::Modules::Morphisms::SparseMatrixMorphismOnDenseVector;

    SetupConjugateGradient() = delete;

    SetupConjugateGradient(
      const HostCompressedSparseRowMatrix& host_csr,
      const std::vector<float>& host_b);

    ~SetupConjugateGradient() = default;

    std::size_t M_;
    CompressedSparseRowMatrix A_;
    DenseVector b_;
    DenseVector x_;
    DenseVector Ax_;
    SparseMatrixMorphismOnDenseVector morphism_;
    Array r_;
    VectorOperations operations_;
    DenseVector p_;
};

} // namespace Solvers
} // namespace Algebra

#endif // ALGEBRA_SOLVERS_SETUP_CONJUGATE_GRADIENT_H