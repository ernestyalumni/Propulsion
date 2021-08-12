#include "Utilities/BinarySearch.h"

#include "gtest/gtest.h"

#include <array>
#include <cstddef> // std::size_t;

using Utilities::Details::calculate_midpoint;
using Utilities::binary_search_for_nearest_left_index;
using std::array;
using std::size_t;

namespace GoogleUnitTests
{
namespace Utilities
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CalculateMidpointTests, ReturnsMidpointIndexForOddNumberOfElements)
{
  EXPECT_EQ(calculate_midpoint(0, 0), 0);
  EXPECT_EQ(calculate_midpoint(0, 2), 1);
  EXPECT_EQ(calculate_midpoint(0, 4), 2);
  EXPECT_EQ(calculate_midpoint(1, 1), 1);
  EXPECT_EQ(calculate_midpoint(1, 3), 2);
  EXPECT_EQ(calculate_midpoint(1, 5), 3);
  EXPECT_EQ(calculate_midpoint(2, 2), 2);
  EXPECT_EQ(calculate_midpoint(2, 4), 3);
  EXPECT_EQ(calculate_midpoint(2, 6), 4);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CalculateMidpointTests, ReturnsMidpointIndexForEvenNumberOfElements)
{
  EXPECT_EQ(calculate_midpoint(0, 1), 0);
  EXPECT_EQ(calculate_midpoint(0, 3), 1);
  EXPECT_EQ(calculate_midpoint(0, 5), 2);
  EXPECT_EQ(calculate_midpoint(1, 2), 1);
  EXPECT_EQ(calculate_midpoint(1, 4), 2);
  EXPECT_EQ(calculate_midpoint(1, 6), 3);
  EXPECT_EQ(calculate_midpoint(2, 3), 2);
  EXPECT_EQ(calculate_midpoint(2, 5), 3);
  EXPECT_EQ(calculate_midpoint(2, 7), 4);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForNearestLeftIndexTests, WorksForNEquals2AndTargetWithin)
{
  constexpr size_t N {2};
  const array<double, N> x {0.25, 8.5};
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 0.5, N), 0);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 0.2578125, N), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForNearestLeftIndexTests, WorksForNEquals2AndTargetOnBoundary)
{
  constexpr size_t N {2};
  const array<double, N> x {0.25, 8.5};
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 0.25, N), 0);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 8.5, N), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForNearestLeftIndexTests, WorksForNEquals3AndTargetWithin)
{
  constexpr size_t N {3};
  const array<double, N> x {0.25, 4.5, 8.3125};
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 0.2578125, N), 0);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 4.25, N), 0);

  EXPECT_EQ(binary_search_for_nearest_left_index(x, 4.625, N), 1);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 8.25, N), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForNearestLeftIndexTests, WorksForNEquals3AndTargetOnBoundary)
{
  constexpr size_t N {3};
  const array<double, N> x {0.25, 4.5, 8.3125};

  EXPECT_EQ(binary_search_for_nearest_left_index(x, 0.25, N), 0);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 4.5, N), 1);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 8.3125, N), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForNearestLeftIndexTests, WorksForNEquals4AndTargetWithin)
{
  constexpr size_t N {4};
  const array<double, N> x {0.25, 4.5, 8.3125, 13.0625};
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 0.2578125, N), 0);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 4.25, N), 0);

  EXPECT_EQ(binary_search_for_nearest_left_index(x, 4.625, N), 1);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 8.25, N), 1);

  EXPECT_EQ(binary_search_for_nearest_left_index(x, 8.5, N), 2);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 13.0, N), 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForNearestLeftIndexTests, WorksForNEquals4AndTargetOnBoundary)
{
  constexpr size_t N {4};
  const array<double, N> x {0.25, 4.5, 8.3125, 13.0625};

  EXPECT_EQ(binary_search_for_nearest_left_index(x, 0.25, N), 0);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 4.5, N), 1);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 8.3125, N), 2);
  EXPECT_EQ(binary_search_for_nearest_left_index(x, 13.0625, N), 2);
}

} // namespace Utilities
} // namespace GoogleUnitTests