#ifndef ALGEBRA_SOLVERS_CONJUGATE_GRADIENT_H
#define ALGEBRA_SOLVERS_CONJUGATE_GRADIENT_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"

#include <cstddef>
#include <optional>

namespace Algebra
{
namespace Solvers
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

    ConjugateGradient() = delete;

    ConjugateGradient(
      CompressedSparseRowMatrix& A,
      const DenseVector& b,
      SparseMatrixMorphismOnDenseVector& morphism,
      CuBLASVectorOperations& vector_operations,
      const std::size_t maximum_iteration=10000,
      const float tolerance=1e-5f);

    ~ConjugateGradient() = default;

    static bool create_default_initial_guess(DenseVector& x);

    std::optional<float> initial_step(
      DenseVector& x,
      DenseVector& Ax,
      Array& r_0);

    //--------------------------------------------------------------------------
    /// \return If has_value(), returns r0, r1.
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    /// \param Ap - you need to provide a DenseVector to store the result of
    /// doing multiplication of matrix A on a vector, x.
    /// \param r - residuals.
    /// \param p - orthonormal direction vectors.
    //--------------------------------------------------------------------------
    std::optional<std::tuple<float, float>> step(
      const std::size_t k,
      const float r0,
      const float r1,
      DenseVector& x,
      Array& r,
      DenseVector& p,
      DenseVector& Ap);

    //--------------------------------------------------------------------------
    /// \returns Is error free and number of iterations done. All inputs x, Ax,
    /// r, p will be mutated.
    //--------------------------------------------------------------------------
    std::tuple<bool, std::size_t> solve(
      DenseVector& x,
      DenseVector& Ax,
      Array& r,
      DenseVector& p);

  private:

    // We chose to have references to objects that we do not expect to mutate at
    // all.

    CompressedSparseRowMatrix& A_;
    const DenseVector& b_;

    SparseMatrixMorphismOnDenseVector& morphism_;
    CuBLASVectorOperations& vector_operations_;

    std::size_t maximum_iteration_;
    float tolerance_;
};

//------------------------------------------------------------------------------
/// \details This class is prefixed with "Sample" because it follows the sample
/// found in cuda-samples from NVIDIA. In developing a conjugate gradient
/// method, the code provided by NVIDIA was used in this class as a sort of
/// baseline. It is not meant to be the "final product" for a conjugate gradient
/// method solver.
//------------------------------------------------------------------------------
class SampleConjugateGradient
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

    SampleConjugateGradient(
      const std::size_t maximum_iteration=10000,
      const float tolerance=1e-5f);

    ~SampleConjugateGradient() = default;

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

} // namespace Solvers
} // namespace Algebra

#endif // ALGEBRA_SOLVERS_CONJUGATE_GRADIENT_H