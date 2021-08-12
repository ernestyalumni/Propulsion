#ifndef UTILITIES_BINARY_SEARCH_H
#define UTILITIES_BINARY_SEARCH_H

#include <array>
#include <cassert>
#include <cstddef> // std::size_t

namespace Utilities
{

namespace Details
{

//------------------------------------------------------------------------------
/// \details Avoid overflow from adding two large numbers.
///
/// If l - r is odd, so that the total number of elements is even, midpoint is
/// the position of the "farthest right" element in the "left" half, i.e. the
/// largest number on the "left, lower" half of the range of numbers.
//------------------------------------------------------------------------------
std::size_t calculate_midpoint(const std::size_t l, const std::size_t r);

} // namespace Details

using BinarySearchBoundaries = std::array<std::size_t, 2>;

//------------------------------------------------------------------------------
/// \details Asserts that for l + 1 < r (so there are at least 3 or more
/// elements to consider), x[l] < x[r].
//------------------------------------------------------------------------------

template <class TContainer>
BinarySearchBoundaries binary_search_for_nearest_iterative(
  const TContainer& x,
  const double target,
  const std::size_t l,
  const std::size_t r)
{
  if (l + 1 >= r)
  {
    return BinarySearchBoundaries{l, r};
  }

  assert(x[l] < x[r]);

  // Midpoint index.
  assert (l < r);
  const std::size_t m {Details::calculate_midpoint(l, r)};
  const double midpoint_value {x[m]};

  if (midpoint_value == target)
  {
    return BinarySearchBoundaries{m, m};
  }

  if (target < midpoint_value)
  {
    return BinarySearchBoundaries{l, m};
  }

  assert (target > midpoint_value);

  return BinarySearchBoundaries{m, r};
}

//------------------------------------------------------------------------------
/// \brief Get the index j such that x[j] <= target and x[j + 1] > target.
/// \param N size of array x.
//------------------------------------------------------------------------------
template <class TContainer>
std::size_t binary_search_for_nearest_left_index(
  const TContainer& x,
  const double target,
  const std::size_t N)
{
  assert (N >= 2);

  std::size_t l {0};
  std::size_t r {N - 1};

  while (l + 1 < r)
  {
    const BinarySearchBoundaries lr {
      binary_search_for_nearest_iterative<TContainer>(x, target, l, r)};

    l = lr[0];
    r = lr[1];
  }

  return l;
}

} // namespace Utilities

#endif // UTILITIES_BINARY_SEARCH_H