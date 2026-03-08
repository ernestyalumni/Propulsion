#include "BinarySearch.h"

#include <cassert>
#include <cstddef> // std::size_t

using std::size_t;

namespace Utilities
{

namespace Details
{

size_t calculate_midpoint(const size_t l, const size_t r)
{
  assert (l <= r);

  const std::size_t L {r - l + 1};

  // See 
  return (L % 2 != 0) ? (L / 2 + l) : (L / 2 - 1 + l);
}

} // namespace Details

} // namespace Utilities
