#include "Arithmetic/IntegerPower.h"
#include "Grid2d.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

using Arithmetic::integer_power;
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

Grid2d::Grid2d(const size_t M, const size_t N):
  values_((M + 2) * (N + 2), 0),
  M_{M},
  N_{N}
{
  // TODO: This step might be redundant.
  values_.reserve((M_ + 2) * (N_ + 2));
}

Grid2d Grid2d::refine(const Grid2d& grid, const int refine)
{
  const uint64_t refinement_factor {
    integer_power(2, static_cast<uint64_t>(refine))};

  Grid2d refined {
    grid.get_M() * refinement_factor,
    grid.get_N() * refinement_factor};

  // Get "inner" values and fill up new grid with those values.
  for (size_t i {0}; i < grid.get_M(); ++i)
  {
    for (size_t j {0}; j < grid.get_N(); ++j)
    {
      const int original_inner_value {grid.at(i + 1, j + 1)};

      for (size_t ii {0}; ii < refinement_factor; ++ii)
      {
        for (size_t jj {0}; jj < refinement_factor; ++jj)
        {
          refined.at(
            i * refinement_factor + ii + 1,
            j * refinement_factor + jj + 1) = original_inner_value;
        }
      }
    }
  }

  // Fill domain boundaries.

  for (size_t i {0}; i < grid.get_M(); ++i)
  {
    for (size_t ii {0}; ii < refinement_factor; ++ii)
    {
      refined.at(i * refinement_factor + ii + 1, 0) = grid.at(i + 1, 0);
      refined.at(
        i * refinement_factor + ii + 1,
        refined.get_N() + 1) = grid.at(i + 1, grid.get_N() + 1);
    }
  }

  for (size_t j {0}; j < grid.get_N(); ++j)
  {
    for (size_t jj {0}; jj < refinement_factor; ++jj)
    {
      refined.at(0, j * refinement_factor + jj + 1) = grid.at(0, j + 1);
      refined.at(
        refined.get_M() + 1,
        j * refinement_factor + jj + 1) = grid.at(grid.get_M() + 1, j + 1);
    }
  }

  refined.at(0, 0) = grid.at(0, 0);
  refined.at(0, refined.get_N() + 1) = grid.at(0, grid.get_N() + 1);
  refined.at(refined.get_M() + 1, 0) = grid.at(grid.get_M() + 1, 0);
  refined.at(refined.get_M() + 1, refined.get_N() + 1) =
    grid.at(grid.get_M() + 1, grid.get_N() + 1);

  return refined;
}

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds