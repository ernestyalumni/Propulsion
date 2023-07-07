#ifndef FIBER_BUNDLES_PGM_GEOMETRY_CELL_H
#define FIBER_BUNDLES_PGM_GEOMETRY_CELL_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace FiberBundles
{
namespace PgmGeometry
{

class Cell
{
  public:

    enum class CellType
    {
      Fluid,
      Outlet,
      Inlet,
      NoSlipWall,
      FreeSlipWall,
      Default
    };

    enum class BorderPosition : uint8_t
    {
      Top = 0,
      Bottom = 1,
      Left = 2,
      Right = 3
    };

    Cell();

    Cell(
      const std::size_t i,
      const std::size_t j,
      const CellType cell_type,
      const std::optional<int> id = std::nullopt);

    ~Cell();

    inline std::shared_ptr<Cell> get_neighbor(
      const BorderPosition border_position)
    {
      return neighbors_[static_cast<uint8_t>(border_position)];
    }

    inline void set_neighbor(
      std::shared_ptr<Cell> cell_ptr,
      const BorderPosition border_position)
    {
      neighbors_[static_cast<uint8_t>(border_position)] = cell_ptr;
    }

    inline std::size_t i() const
    {
      return i_;
    }

    inline std::size_t j() const
    {
      return j_;
    }

    inline CellType get_type() const
    {
      return type_;
    }

    inline std::optional<int> get_id() const
    {
      return id_;
    }

  private:

    // Pointers to neighbors: TOP - BOTTOM - LEFT - RIGHT - NORTHWEST -
    // SOUTHEAST
    std::array<std::shared_ptr<Cell>, 6> neighbors_;

    std::array<bool, 4> is_border_position_;

    // x index
    std::size_t i_;
    // y index
    std::size_t j_;

    CellType type_;

    // Cell ID (only necessary for walls)
    std::optional<int> id_;
};

} // namespace PgmGeometry

} // namespace FiberBundles

#endif // FIBER_BUNDLES_PGM_GEOMETRY_CELL_H