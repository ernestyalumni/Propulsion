#include "BiconjugateGradientStabilized.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <tuple> // std::get
#include <vector>

using std::get;
using std::make_optional;
using std::make_tuple;
using std::nullopt;
using std::optional;
using std::tuple;

namespace Algebra
{
namespace Solvers
{

BiconjugateGradientStabilized::BiconjugateGradientStabilized(
  CompressedSparseRowMatrix& A,
  const DenseVector& b,
  SparseMatrixMorphismOnDenseVector& morphism,
  CuBLASVectorOperations& vector_operations,
  const std::size_t maximum_iteration,
  const double tolerance
  ):
  A_{A},
  b_{b},
  rtilde_0_{b.number_of_elements_},
  morphism_{morphism},
  vector_operations_{vector_operations},
  maximum_iteration_{maximum_iteration},
  tolerance_{tolerance}
{
  assert(morphism_.get_buffer_size() > 0);
}

bool BiconjugateGradientStabilized::create_default_initial_guess(DenseVector& x)
{
  std::vector<double> h_x (x.number_of_elements_, 0.0);
  return x.copy_host_input_to_device(h_x);
}

bool BiconjugateGradientStabilized::initial_step(
  DenseVector& x_0,
  DenseVector& Ax,
  Array& r_0,
  DenseVector& p_0,
  std::optional<Array>& rtilde_0_guess)
{
  // Ax = A * x; r_0 := b
  if (!morphism_.linear_transform(A_, x_0, Ax) || !vector_operations_.copy(b_, r_0))
  {
    return false;
  }

  // r_0 := (-1.0) * Ax + b
  if (!vector_operations_.scalar_multiply_and_add_vector(-1.0, Ax, r_0))
  {
    return false;
  }

  if (rtilde_0_guess.has_value())
  {
    const auto result = vector_operations_.dot_product(r_0, *rtilde_0_guess);

    if (!result.has_value())
    {
      return false;
    }

    vector_operations_.copy(
      (*result > tolerance_ ? *rtilde_0_guess : r_0),
      rtilde_0_);
  }
  else if (!vector_operations_.copy(r_0, rtilde_0_))
  {
    return false;
  }

  if (!vector_operations_.copy(r_0, p_0))
  {
    return false;
  }

  return true;
}

optional<tuple<bool, double, double, double>> BiconjugateGradientStabilized::
  step(
  DenseVector& x,
  Array& r,
  DenseVector& p,
  DenseVector& Ap,
  DenseVector& s)
{
  // Ap := A p
  if (!morphism_.linear_transform(A_, p, Ap))
  {
    return nullopt;
  }

  // (r_0' \cdot r_j) \equiv (r_0' * r_j)
  const auto rtilde_0_dot_r_j = vector_operations_.dot_product(rtilde_0_, r);
  // (r_0' \cdot Ap_j) \equiv (r_0' * A p_j)
  const auto rtilde_0_dot_Ap = vector_operations_.dot_product(rtilde_0_, Ap);

  if (!rtilde_0_dot_Ap.has_value() || !rtilde_0_dot_r_j.has_value())
  {
    return nullopt;
  }

  // \alpha_j = (r_0' * r_j) / (r_0' * Ap_j)
  const double alpha_j {*rtilde_0_dot_r_j / *rtilde_0_dot_Ap};

  // s := r
  vector_operations_.copy(r, s);

  // s_{j} := - \alpha_j Ap_j + s_j = -\alpha_j Ap_j + r_j
  if (!vector_operations_.scalar_multiply_and_add_vector(-alpha_j, Ap, s))
  {
    return nullopt;
  }

  const auto s_norm = vector_operations_.get_norm(s);
  if (!s_norm.has_value())
  {
    return nullopt;
  }
  if (*s_norm < tolerance_)
  {
    if (!vector_operations_.scalar_multiply_and_add_vector(alpha_j, p, x))
    {
      return nullopt;
    }

    return make_optional(make_tuple(false, alpha_j, -1.0, -1.0));
  }

  // Reuse Ap as an output "container" to store the value of the computation of
  // As \equiv A * s.

  // As := A s
  if (!morphism_.linear_transform(A_, s, Ap))
  {
    return nullopt;
  }

  // (s_j \cdot As_j) \equiv (s_j * A s_j)
  const auto s_dot_As {vector_operations_.dot_product(s, Ap)};
  // (As_j \cdot As_j) \equiv (As_j * As_j)
  const auto As_dot_As {vector_operations_.dot_product(Ap, Ap)};

  if (!s_dot_As.has_value() || !As_dot_As.has_value())
  {
    return nullopt;
  }

  const double omega_j {*s_dot_As / *As_dot_As};

  // x_{j+1} = x_j + \alpha_j p_j + \omega_j s_j
  if (!vector_operations_.scalar_multiply_and_add_vector(alpha_j, p, x) ||
    !vector_operations_.scalar_multiply_and_add_vector(omega_j, s, x))
  {
    return nullopt;
  }

  // r := s
  vector_operations_.copy(s, r);

  // r_{j+1} = s_j - \omega_j As_j
  if (!vector_operations_.scalar_multiply_and_add_vector(-omega_j, Ap, r))
  {
    return nullopt;
  }

  const auto r_norm = vector_operations_.get_norm(r);
  if (!r_norm.has_value())
  {
    return nullopt;
  }
  if (*r_norm < tolerance_)
  {
    return make_optional(make_tuple(false, alpha_j, -1.0, omega_j));
  }

  // (r_0' * r_{j+1})
  const auto rtilde_0_dot_r_jp1 = vector_operations_.dot_product(rtilde_0_, r);
  if (!rtilde_0_dot_r_jp1.has_value())
  {
    return nullopt;
  }

  const double beta_j {
    alpha_j / omega_j * *rtilde_0_dot_r_jp1 / *rtilde_0_dot_r_j};

  // Ap := A * p
  if (!morphism_.linear_transform(A_, p, Ap))
  {
    return nullopt;
  }

  if (!vector_operations_.scalar_multiply_and_add_vector(-omega_j, Ap, p))
  {
    return nullopt;
  }

  if (!vector_operations_.scalar_multiply(beta_j, p))
  {
    return nullopt;
  }

  if (!vector_operations_.scalar_multiply_and_add_vector(1.0, r, p))
  {
    return nullopt;
  }

  if (*rtilde_0_dot_r_jp1 < tolerance_)
  {
    // r_0~ = r_{j+1}
    vector_operations_.copy(r, rtilde_0_);
    // p_{j+1} = r_{j+1}
    vector_operations_.copy(r, p);
  }

  return make_optional(make_tuple(true, alpha_j, beta_j, omega_j));
}

tuple<bool, std::size_t> BiconjugateGradientStabilized::solve(
  DenseVector& x,
  DenseVector& Ax,
  Array& r,
  DenseVector& p,
  DenseVector& s)
{
  optional<Array> no_rtilde_0_guess {nullopt};

  if (!initial_step(x, Ax, r, p, no_rtilde_0_guess))
  {
    return make_tuple(false, 0);
  }

  std::size_t k {0};

  for (; k < maximum_iteration_; ++k)
  {
    const auto step_result = step(x, r, p, Ax, s);

    if (!step_result.has_value())
    {
      return make_tuple(false, k);
    }

    if (!get<0>(*step_result))
    {
      return make_tuple(true, k);
    }
  }

  return make_tuple(true, k);
}

} // namespace Solvers
} // namespace Algebra