#ifndef ALGEBRA_SOLVERS_BICONJUGATE_GRADIENT_STABILIZED_H
#define ALGEBRA_SOLVERS_BICONJUGATE_GRADIENT_STABILIZED_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"

#include <optional>

namespace Algebra
{
namespace Solvers
{

class BiconjugateGradientStabilized
{
  public:

    using Array = Algebra::Modules::Vectors::DoubleArray;

    using CompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::DoubleCompressedSparseRowMatrix;

    using CuBLASVectorOperations =
      Algebra::Modules::Vectors::DoubleCuBLASVectorOperations;

    using DenseVector =
      Algebra::Modules::Matrices::SparseMatrices::DoubleDenseVector;

    using SparseMatrixMorphismOnDenseVector =
      Algebra::Modules::Morphisms::DoubleSparseMatrixMorphismOnDenseVector;

    BiconjugateGradientStabilized() = delete;

    BiconjugateGradientStabilized(
      CompressedSparseRowMatrix& A,
      const DenseVector& b,
      SparseMatrixMorphismOnDenseVector& morphism,
      CuBLASVectorOperations& vector_operations,
      const std::size_t maximum_iteration=10000,
      const double tolerance=1e-6);

    ~BiconjugateGradientStabilized() = default;

    static bool create_default_initial_guess(DenseVector& x);

    //--------------------------------------------------------------------------
    /// \details Does the following:
    /// Given initial guess x_0 as an input,
    /// Compute r_0 = b - Ax_0
    /// Chooses a r_0~ s.t. r_0~ * r_0 != 0; picks r_0~ = r_0 if not given
    /// a guess, input.
    /// Sets p_0 = r_0    
    //--------------------------------------------------------------------------
    bool initial_step(
      DenseVector& x_0,
      DenseVector& Ax,
      Array& r_0,
      DenseVector& p_0,
      std::optional<Array>& rtilde_0_guess);

    //--------------------------------------------------------------------------
    /// \param Ap - you need to provide a DenseVector to store the result of
    /// doing multiplication of matrix A on a vector, x.
    /// \param r - residuals.
    /// \param p - orthonormal direction vectors.
    /// \returns Is parent for loop meant to continue, i.e. is continue? and
    /// values for alpha, beta, omega.
    //--------------------------------------------------------------------------
    std::optional<std::tuple<bool, double, double, double>> step(
      DenseVector& x,
      Array& r,
      DenseVector& p,
      DenseVector& Ap,
      DenseVector& s);

    std::tuple<bool, std::size_t> solve(
      DenseVector& x,
      DenseVector& Ax,
      Array& r,
      DenseVector& p,
      DenseVector& s);

  private:

    // We chose to have references to objects that we do not expect to mutate at
    // all.

    CompressedSparseRowMatrix& A_;
    const DenseVector& b_;

    // In https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
    // this is \mathbf{r}_i^{\tilde} or the "stabilized" residual, i.e.
    // r_i^~ = Q_i(A)P_i(A) r_0
    // where Q_i, P_i are polynomials in A.
    DenseVector rtilde_0_;

    SparseMatrixMorphismOnDenseVector& morphism_;
    CuBLASVectorOperations& vector_operations_;

    std::size_t maximum_iteration_;
    double tolerance_;
};

} // namespace Solvers
} // namespace Algebra

#endif // ALGEBRA_SOLVERS_BICONJUGATE_GRADIENT_STABILIZED_H