#ifndef ALGEBRA_MODULES_OPTIMIZATION_CONJUGATE_GRADIENT_H
#define ALGEBRA_MODULES_OPTIMIZATION_CONJUGATE_GRADIENT_H

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"

#include <cstddef>

namespace Algebra
{
namespace Modules
{
namespace Optimization
{

class ConjugateGradient
{
  public:

    using CompressedSparseRowMatrix =
      Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;

    ConjugateGradient(
      const std::size_t maximum_iteration=10000,
      const float tolerance=1e-5f);

    ~ConjugateGradient() = default;

    void initialize();

}

} // namespace Optimization
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_OPTIMIZATION_CONJUGATE_GRADIENT_H