#include "ExponentiationBySquaring.h"

#include <cstdint>

namespace Numerical
{

uint64_t exponentiation_by_squaring_iterative(const uint64_t x, const uint32_t n)
{
  if (n == 0)
  {
    return 1;
  }

  uint32_t N {n};
  uint64_t y {1};
  uint64_t b {x};
  while (N > 1)
  {
    // If n even.
    if (N % 2 == 0)
    {
      b = b * b;
      N = N / 2;
    }
    else
    {
      y = b * y;
      b = b * b;
      N = (N - 1) / 2;
    }
  }

  return x * y;
}

} // namespace Numerical