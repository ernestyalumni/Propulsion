#ifndef MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_GRID2D_METRIC_DATA_H
#define MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_GRID2D_METRIC_DATA_H

#include <cstddef>

namespace Manifolds
{
namespace Euclidean
{
// Pgm refers to the .pgm suffix used for the "configuration" files here,
// https://github.com/yuphin/turbulent-cfd/tree/master/example_cases
// within each example case.
namespace PgmGeometry
{

//------------------------------------------------------------------------------
/// \details For smooth manifolds, defining a (Riemannian) metric completely
/// defines the notion of "distance" and "length" on a manifold.
//------------------------------------------------------------------------------
struct Grid2dMetricData
{
  // Values for convenience.

  // Minimum value of index i (in x-direction)
  std::size_t minimum_i_ {0};
  // Maximum value of index i (in x-direction)
  std::size_t maximum_i_ {0};

  // Minimum value of index j (in x-direction)
  std::size_t minimum_j_ {0};
  // Maximum value of index j (in x-direction)
  std::size_t maximum_j_ {0};

  // Number of inner x divisions, excluding each of 2 cells at the end.
  std::size_t M_ {0};
  // Number of inner y divisions, excluding each of 2 cells at the end.
  std::size_t N_ {0};

  // Total number of cells, including the boundaries.
  std::size_t total_number_of_cells_;

  double x_length_;
  double y_length_;

  double dx_;
  double dy_;
};

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds

#endif // MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_GRID2D_METRIC_DATA_H