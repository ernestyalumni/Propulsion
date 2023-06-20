#ifndef ARITHMETIC_INTEGER_POWER_H
#define ARITHMETIC_INTEGER_POWER_H

#include <cstdint> // uint64_t

namespace Arithmetic
{

//------------------------------------------------------------------------------
/// \details This was prompted by this article, pow() is bad!
/// https://codeforces.com/blog/entry/87935
/// Making this a constexpr function require it to be completely in the header
/// for the compiler to see:
/// https://stackoverflow.com/questions/27345284/is-it-possible-to-declare-constexpr-class-in-a-header-and-define-it-in-a-separat
//------------------------------------------------------------------------------
constexpr uint64_t integer_power(uint64_t base, uint64_t exp)
{
  uint64_t result {1};

  while (exp)
  {
    if (exp & 1)
    {
      result *= base;
    }
    base *= base;

    exp >>= 1;
  }

  return result;
}

} // namespace Arithmetic

#endif // ARITHMETIC_INTEGER_POWER_H