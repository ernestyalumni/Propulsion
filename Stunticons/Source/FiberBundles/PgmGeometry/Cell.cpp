#include "Cell.h"

#include <cstddef>
#include <optional>

using std::nullopt;
using std::size_t;

namespace FiberBundles
{
namespace PgmGeometry
{

Cell::Cell():
  closest_distance_{nullopt},
  closest_wall_index_{nullopt},
  borders_{},
  neighbors_{},
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
  closest_distance_{nullopt},
  closest_wall_index_{nullopt},
  borders_{},
  neighbors_{},
  is_border_position_{false, false, false, false},
  i_{i},
  j_{j},
  type_{cell_type},
  id_{id}
{}

void Cell::add_border(const BorderPosition border_position)
{
  is_border_position_[static_cast<uint8_t>(border_position)] = true;
  borders_.push_back(border_position);
}

} // namespace PgmGeometry
} // namespace FiberBundles