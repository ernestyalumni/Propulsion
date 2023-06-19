#include "Grid2d.h"

#include <concepts>
#include <cstddef>
#include <vector>

namespace Manifolds
{
namespace Euclidean
{
// Pgm refers to the .pgm suffix used for the "configuration" files here,
// https://github.com/yuphin/turbulent-cfd/tree/master/example_cases
// within each example case.
namespace PgmGeometry
{

Grid2d::Grid2d(const std::size_t M, const std::size_t N):
  values_((M + 2) * (N + 2), 0),
  M_{M},
  N_{N}
{
  // TODO: This step might be redundant.
  values_.reserve((M_ + 2) * (N_ + 2));
}

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds