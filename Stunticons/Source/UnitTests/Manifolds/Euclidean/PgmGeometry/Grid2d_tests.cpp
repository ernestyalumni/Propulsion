#include "Manifolds/Euclidean/PgmGeometry/Grid2d.h"
#include "gtest/gtest.h"

#include <cstddef> // std::size_t

using Manifolds::Euclidean::PgmGeometry::Grid2d;
using std::size_t;

namespace GoogleUnitTests
{

namespace Manifolds
{
namespace Euclidean
{
namespace PgmGeometry
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Grid2dTests, Constructible)
{
  Grid2d gr {6, 9};

  for (size_t i {0}; i < gr.get_M() + 2; ++i)
  {
    for (size_t j {0}; j < gr.get_N() + 2; ++j)
    {
      EXPECT_EQ(gr.get(i, j), 0);
      EXPECT_EQ(gr.at(i, j), 0);
    }
  }
}

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds

} // namespace GoogleUnitTests