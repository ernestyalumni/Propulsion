#include "Cell.h"

#include <cstddef>
#include <optional>

using std::size_t;

namespace FiberBundles
{
namespace PgmGeometry
{

Cell::Cell():

  is_border_position_{false, false, false, false},
  i_{0},
  j_{0},
  type_{CellType::Default},
  id_{std::nullopt}
{}

Cell::Cell(
  const size_t i,
  const size_t j,
  const CellType cell_type,
  const std::optional<int> id
  ):
  is_border_position_{false, false, false, false},
  i_{i},
  j_{j},
  type_{cell_type},
  id_{id}
{}

} // namespace PgmGeometry
} // namespace FiberBundles