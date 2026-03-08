#include "Numerical/Interpolation/Interpolation.h"

#include "gtest/gtest.h"

#include <vector>

using Numerical::Interpolation::Details::binary_search_for_subrange_left_edge;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace Interpolation
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForSubrangeLeftEdgeTests, WorksForNEquals2MEquals2WithinBounds)
{
  const vector<double> x {0.25, 8.5};
  const size_t N {x.size()};
  const size_t M {2};

  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 0.5, N, M), 0);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 0.2578125, N, M), 0);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 8.25, N, M), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForSubrangeLeftEdgeTests, WorksForNEquals2MEquals2OnBounds)
{
  const vector<double> x {0.25, 8.5};
  const size_t N {x.size()};
  const size_t M {2};

  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 0.25, N, M), 0);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 8.5, N, M), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForSubrangeLeftEdgeTests, WorksForNEquals3MEquals2WithinBounds)
{
  const vector<double> x {0.25, 4.5, 8.3125};
  const size_t N {x.size()};
  const size_t M {2};

  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 0.2576125, N, M), 0);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 4.25, N, M), 0);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 4.625, N, M), 1);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 8.25, N, M), 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BinarySearchForSubrangeLeftEdgeTests, WorksForNEquals3MEquals2OnBounds)
{
  const vector<double> x {0.25, 4.5, 8.3125};
  const size_t N {x.size()};
  const size_t M {2};

  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 0.25, N, M), 0);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 4.5, N, M), 1);
  EXPECT_EQ(binary_search_for_subrange_left_edge(x, 8.3125, N, M), 1);
}

} // namespace Interpolation
} // namespace Numerical
} // namespace GoogleUnitTests