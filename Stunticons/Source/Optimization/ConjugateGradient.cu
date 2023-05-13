#include "ConjugateGradient.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <tuple>

using std::nullopt;
using std::optional;

namespace Optimization
{

ConjugateGradient::ConjugateGradient(
  const std::size_t maximum_iteration,
  const float tolerance
  ):
  maximum_iteration_{maximum_iteration},
  tolerance_{tolerance}
{}

bool ConjugateGradient::create_default_initial_guess(DenseVector& x)
{
  std::vector h_x (x.number_of_elements_, 0.0f);
  return x.copy_host_input_to_device(h_x);
}

optional<float> ConjugateGradient::initial_step(
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

optional<std::tuple<float, float>> ConjugateGradient::step(
  const std::size_t k,
  const float r0,
  const float r1,
  DenseVector& p,
  Array& r,
  SparseMatrixMorphismOnDenseVector& morphism,
  CuBLASVectorOperations& vector_operations,
  CompressedSparseRowMatrix& A,
  DenseVector& Ax,
  DenseVector& x)
{
  if (k > 1)
  {
    const float b {r1 / r0};
    vector_operations.scalar_multiply(b, p);
    vector_operations.scalar_multiply_and_add_vector(1.0, r, p);
  }
  else
  {
    vector_operations.copy(r, p);
  }

  if (!morphism.linear_transform(A, p, Ax))
  {
    return nullopt;
  }

  const auto p_dot_Ax {vector_operations.dot_product(p, Ax)};
  if (!p_dot_Ax.has_value())
  {
    return nullopt;
  }

  const float a {r1 / *p_dot_Ax};

  if (!vector_operations.scalar_multiply_and_add_vector(a, p, x))
  {
    return nullopt;
  }

  if (!vector_operations.scalar_multiply_and_add_vector(-a, Ax, r))
  {
    return nullopt;
  }

  const float new_r0 {r1};

  const auto r1_squared = vector_operations.dot_product(r, r);

  if (!r1_squared.has_value())
  {
    return nullopt;
  }

  cudaDeviceSynchronize();

  return std::make_optional(std::make_tuple(new_r0, *r1_squared));
}


} // namespace Optimization