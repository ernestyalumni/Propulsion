#include "Grid2d.h"
#include "initialize_lid_driven_cavity.h"

#include <cstddef> // std::size_t

using std::size_t;

namespace Manifolds
{
namespace Euclidean
{
// Pgm refers to the .pgm suffix used for the "configuration" files here,
// https://github.com/yuphin/turbulent-cfd/tree/master/example_cases
// within each example case.
namespace PgmGeometry
{

void initialize_lid_driven_cavity(Grid2d& grid)
{
  // PGM convention - 11: moving wall
  static constexpr int moving_wall_id {11};

  // PGM convention - 10: fixed wall wall
  static constexpr int fixed_wall_id {10};

  for (size_t i {0}; i < grid.get_M() + 2; ++i)
  {
    for (size_t j {0}; j < grid.get_N() + 2; ++j)
    {
      // Bottom, left and right walls; no-slip
      if (i == 0 || j == 0 || i == grid.get_M() + 1)
      {
        grid.at(i, j) = fixed_wall_id;
      }

      // Top wall: moving wall
      else if (j == grid.get_N() + 1)
      {
        grid.at(i, j) = moving_wall_id;
      }
    }
  }
}

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds