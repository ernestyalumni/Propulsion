#include "Manifolds/Euclidean/PgmGeometry/Grid2d.h"
#include "Manifolds/Euclidean/PgmGeometry/initialize_lid_driven_cavity.h"
#include "gtest/gtest.h"

#include <cstddef> // std::size_t

using Manifolds::Euclidean::PgmGeometry::Grid2d;
using Manifolds::Euclidean::PgmGeometry::initialize_lid_driven_cavity;
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Grid2dTests, RefineCopiesValuesFromOriginalGrid)
{
  Grid2d gr {6, 9};

  initialize_lid_driven_cavity(gr);

  Grid2d refined {Grid2d::refine(gr, 3)};

  EXPECT_EQ(refined.get_M(), 48);
  EXPECT_EQ(refined.get_N(), 72);

  EXPECT_EQ(refined.get(0, 0), gr.get(0, 0));
  EXPECT_EQ(refined.get(0, 73), gr.get(0, 10));
  EXPECT_EQ(refined.get(49, 0), gr.get(7, 0));
  EXPECT_EQ(refined.get(49, 73), gr.get(7, 10));

  for (size_t i {0}; i < refined.get_M() + 2; ++i)
  {
    EXPECT_EQ(refined.get(i, 0), 10);

    if (i == 0 || i == refined.get_M() + 1)
    {
      EXPECT_EQ(refined.get(i, refined.get_N() + 1), 10);
    }
    else
    {
      EXPECT_EQ(refined.get(i, refined.get_N() + 1), 11);      
    }
  }

  for (size_t j {0}; j < refined.get_N() + 2; ++j)
  {
    EXPECT_EQ(refined.get(0, j), 10);
    EXPECT_EQ(refined.get(refined.get_M() + 1, j), 10);
  }
}

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds

} // namespace GoogleUnitTests