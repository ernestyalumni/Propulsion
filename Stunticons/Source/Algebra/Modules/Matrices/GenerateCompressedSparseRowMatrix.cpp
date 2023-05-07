#include "HostCompressedSparseRow.h"

#include <algorithm> // std::min
#include <cstddef> // std::size_t
#include <cstdlib> // std::rand, RAND_MAX

using std::rand;
using std::size_t;

namespace Algebra
{
namespace Modules
{
namespace Matrices
{
namespace SparseMatrices
{

void generate_tridiagonal_matrix(HostCompressedSparseRowMatrix& a)
{
  a.I_[0] = 0;
  a.J_[0] = 0;
  a.J_[1] = 1;

  const std::size_t minimal_N {std::min(a.M_, a.N_)};

  a.values_[0] = static_cast<float>(rand()) / RAND_MAX + 10.0f;
  a.values_[1] = static_cast<float>(rand()) / RAND_MAX;

  for (size_t i {1}; i < minimal_N; ++i)
  {
    if (i > 1)
    {
      a.I_[i] = a.I_[i - 1] + 3;
    }
    else
    {
      a.I_[1] = 2;
    }

    const size_t start {(i - 1) * 3 + 2};
    a.J_[start] = i - 1;
    a.J_[start + 1] = i;

    a.values_[start] = a.values_[start - 1];
    a.values_[start + 1] = static_cast<float>(rand()) / RAND_MAX + 10.0f;

    if (i < minimal_N - 1)
    {
      a.J_[start + 2] = i + 1;

      a.values_[start + 2] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  if (a.M_ > a.N_)
  {
    for (size_t i {a.N_}; i < a.M_; ++i)
    {
      a.I_[i] = a.I_[i - 1];
    }
  }

  a.I_[a.M_] = a.number_of_elements_;
}

} // namespace SparseMatrices

} // namespace Matrices
} // namespace Modules
} // namespace Algebra
