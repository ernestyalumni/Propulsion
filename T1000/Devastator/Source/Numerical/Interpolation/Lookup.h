//------------------------------------------------------------------------------
/// \ref 3.1 Preliminaries: Searching an Ordered Table, pp. 114, Numerical
/// Recipes, 3rd. Ed.
//------------------------------------------------------------------------------

#ifndef NUMERICAL_INTERPOLATION_LOOKUP_H
#define NUMERICAL_INTERPOLATION_LOOKUP_H

#include "Utilities/BinarySearch.h"

#include <algorithm> // std::min
#include <cassert>
#include <cstddef> // std::size_t

namespace Numerical
{
namespace Interpolation
{
namespace Details
{

//------------------------------------------------------------------------------
/// \brief Give a target value, return a value j_lo such that the target value,
///   insofar as possible, is centered in the subrange x[j_lo.. j_lo + M - 1],
///   where x is a (sequential) array of abscissas. The values in x are assumed
///   to be monotonically increasing. Returned value is not less than 0, nor
///   greater than N - M.
//------------------------------------------------------------------------------
template <class TContainer>
std::size_t binary_search_for_subrange_left_edge(
  const TContainer& x,
  const double target,
  const std::size_t N,
  const std::size_t M)
{
  assert(N >= M && N >= 2 && M >= 2);

  const std::size_t m {
    Utilities::binary_search_for_nearest_left_index(x, target, N)};

  return std::min(N - M, m - ((M - 2) >> 1));
}

template <class TContainer>
std::size_t hunt(
  const TContainer& x,
  const double target,
  const std::size_t N,
  const std::size_t M,
  const std::size_t guess_index)
{
  std::size_t l {guess_index};
  std::size_t increment {1};

  assert(N >= M && N >= 2 && M >= 2);

  // Hunt up.
  if (target >= x[l])
  {
    while (true)
    {
      r = l + increment;

      // Off end of table
      if (r >= N - 1)
      {
        r = N - 1;
        break;
      }
      // Found bracket.
      else if (target < x[r])
      {
        break;
      }
      // Not done, so double the increment and try again.
      else
      {
        l = r;
        increment += increment;
      }
    }
  }
  // Hunt down
  else
  {
    r = l;
    while (true)
    {
      // Off end of table.
      if (l <= increment)
      {
        l = 0;
        break;
      }

      l = l - increment;

      // Found bracket.
      if (target >= x[l])
      {
        break;
      }
      // No done, so double the increment and try again.
      else
      {
        r = l;
        increment += increment;
      }
    }
  }

  // Hunt is done, so begin final bisection phase:
  const binary_search_for_subrange_left_edge(x, target, N, M)
}

} // namespace Details
} // namespace Interpolation
} // namespace Numerical

#endif // NUMERICAL_INTERPOLATION_INTERPOLATION_H
