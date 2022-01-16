#ifndef NUMERICAL_EXPONENTIATION_BY_SQUARING_H
#define NUMERICAL_EXPONENTIATION_BY_SQUARING_H

#include <cstdint>

namespace Numerical
{

//------------------------------------------------------------------------------
/// \ref https://en.wikipedia.org/wiki/Exponentiation_by_squaring
//------------------------------------------------------------------------------  
uint64_t exponentiation_by_squaring_iterative(
  const uint64_t x,
  const uint32_t n);

} // namespace Numerical

#endif // NUMERICAL_EXPONENTIATION_BY_SQUARING_H
