#include "ConjugateGradient.h"

#include <cstddef>
#include <optional>

using std::nullopt;
using std::optional;

namespace Algebra
{
namespace Modules
{
namespace Optimization
{

ConjugateGradient::ConjugateGradient(
  const std::size_t maximum_iteration,
  const float tolerance
  ):
  maximum_iteration_{maximum_iteration},
  tolerance_{tolerance}
{}

optional<float> ConjugateGradient::initial_guess(
  SparseMatrixMorphismOnDenseVector& morphism,
  CuBLASVectorOperations& vector_operations,
  CompressedSparseRowMatrix& A,
  DenseVector& x,
  DenseVector& Ax,
  Array& r_0)
{
  if (!morphism.linear_transform(A, x, Ax))
  {
    return nullopt;
  }

  if (!vector_operations.scalar_multiply_and_add_vector(-1.0, Ax, r_0))
  {
    return nullopt;
  }

  return vector_operations.dot_product(r_0, r_0);
}

} // namespace Optimization
} // namespace Modules
} // namespace Algebra