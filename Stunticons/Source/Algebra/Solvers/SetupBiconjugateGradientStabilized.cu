#include "SetupBiconjugateGradientStabilized.h"

#include <vector>

namespace Algebra
{
namespace Solvers
{

SetupBiconjugateGradientStabilized::SetupBiconjugateGradientStabilized(
  const HostCompressedSparseRowMatrix& host_csr,
  const std::vector<double>& host_b
  ):
  M_{host_csr.M_},
  A_{host_csr.M_, host_csr.N_, host_csr.number_of_elements_},
  b_{host_csr.M_},
  x_{host_csr.M_},
  Ax_{host_csr.M_},
  morphism_{},
  r_{host_csr.M_},
  operations_{},
  p_{host_csr.M_},
  s_{host_csr.M_}
{
  A_.copy_host_input_to_device(host_csr);
  morphism_.buffer_size(A_, x_, Ax_);
  b_.copy_host_input_to_device(host_b);
}

} // namespace Solvers
} // namespace Algebra