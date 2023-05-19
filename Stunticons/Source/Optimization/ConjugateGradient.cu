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
  // Ax = A * x
  if (!morphism.linear_transform(A, x, Ax))
  {
    return nullopt;
  }

  // r_0 := (-1.0) * Ax + r_0
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
    // \beta_k = r_k^2 / r^2_{k-1}
    const float b {r1 / r0};
    // \beta * p
    vector_operations.scalar_multiply(b, p);
    // p := (1.0) * r + \beta * p
    vector_operations.scalar_multiply_and_add_vector(1.0, r, p);
  }
  else
  {
    // p := r
    vector_operations.copy(r, p);
  }

  // Ax := A p
  if (!morphism.linear_transform(A, p, Ax))
  {
    return nullopt;
  }

  // Ax := p^T Ap
  const auto p_dot_Ap {vector_operations.dot_product(p, Ax)};
  if (!p_dot_Ap.has_value())
  {
    return nullopt;
  }

  // \alpha = r_k^2 / (p_k^T Ap_K)
  const float a {r1 / *p_dot_Ap};

  // x_{k + 1} := \alpha_k p_k + x_k
  if (!vector_operations.scalar_multiply_and_add_vector(a, p, x))
  {
    return nullopt;
  }

  // r_{k+1} := - \alpha_k Ap_k + r_k
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

bool ConjugateGradient::solve(
  DenseVector& p,
  Array& r,
  SparseMatrixMorphismOnDenseVector& morphism,
  CuBLASVectorOperations& vector_operations,
  CompressedSparseRowMatrix& A,
  DenseVector& Ax,
  DenseVector& x)
{
  const auto r_0_sqrt = initial_step(morphism, vector_operations, A, x, Ax, r);

  if (!r_0_sqrt.has_value())
  {
    return false;
  }

  std::size_t k {1};

  float r_1 {*r_0_sqrt};
  float r_0 {0.0f};

  while (k <= maximum_iteration_ && r_1 > tolerance_ * tolerance_)
  {
    const auto step_results =
      step(k, r_0, r_1, p, r, morphism, vector_operations, A, Ax, x);

    if (!step_results.has_value())
    {
      return false;
    }

    r_0 = std::get<0>(*step_results);
    r_1 = std::get<1>(*step_results);

    ++k;
  }

  return true;
}

} // namespace Optimization