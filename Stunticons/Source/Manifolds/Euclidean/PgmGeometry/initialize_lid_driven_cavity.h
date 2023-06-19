#ifndef MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_INITIALIZE_LID_DRIVEN_CAVITY_H
#define MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_INITIALIZE_LID_DRIVEN_CAVITY_H

#include "Grid2d.h"

#include <cstddef> // std::size_t

namespace Manifolds
{
namespace Euclidean
{
// Pgm refers to the .pgm suffix used for the "configuration" files here,
// https://github.com/yuphin/turbulent-cfd/tree/master/example_cases
// within each example case.
namespace PgmGeometry
{

void initialize_lid_driven_cavity(Grid2d& grid);

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds

#endif // MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_INITIALIZE_LID_DRIVEN_CAVITY_H