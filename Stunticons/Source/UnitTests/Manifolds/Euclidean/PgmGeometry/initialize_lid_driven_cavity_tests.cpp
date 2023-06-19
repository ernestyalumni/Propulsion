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
TEST(InitializeLidDrivenCavityTests, Constructible)
{
  Grid2d gr {6, 9};

  initialize_lid_driven_cavity(gr);

  for (size_t i {0}; i < gr.get_M() + 2; ++i)
  {
    for (size_t j {0}; j < gr.get_N() + 2; ++j)
    {
      // Bottom, left, and right walls; no slip
      if (i == 0 || j == 0 || i == gr.get_M() + 1)
      {
        EXPECT_EQ(gr.get(i, j), 10);
      }
      // Top wall; moving wall
      else if (j == gr.get_N() + 1)
      {
        EXPECT_EQ(gr.get(i, j), 11);
      }
      else
      {
        EXPECT_EQ(gr.get(i, j), 0);
        EXPECT_EQ(gr.at(i, j), 0);
      }
    }
  }
}

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds

} // namespace GoogleUnitTests