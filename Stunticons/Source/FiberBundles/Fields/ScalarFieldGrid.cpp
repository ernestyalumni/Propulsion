#include "ScalarFieldGrid.h"

#include <cstddef>

using std::size_t;

namespace FiberBundles
{
namespace Fields
{

ScalarFieldGrid::ScalarFieldGrid(
  const size_t M,
  const size_t N,
  const double initial_value):
  values_(M * N, initial_value),
  M_{M},
  N_{N}
{
  // TODO: This step might be redundant.
  values_.reserve(M * N); 
}

} // namespace Fields
} // namespace FiberBundles