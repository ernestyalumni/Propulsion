#ifndef FIBER_BUNDLES_PGM_GEOMETRY_CELL_GRID_H
#define FIBER_BUNDLES_PGM_GEOMETRY_CELL_GRID_H

#include "Cell.h"
#include "Manifolds/Euclidean/PgmGeometry/Grid2d.h"
#include "Manifolds/Euclidean/PgmGeometry/Grid2dMetricData.h"

#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

namespace FiberBundles
{

namespace PgmGeometry
{

class CellGrid
{
  public:

    class Grid
    {
      public:

        Grid() = delete;

        //----------------------------------------------------------------------
        /// \details Create a total of (M + 2) * (N + 2) Cell type elements.
        //----------------------------------------------------------------------
        Grid(const std::size_t M, const std::size_t N);

        inline Cell& operator()(const std::size_t i, const std::size_t j)
        {
          return elements_[i + j * I_];
        }

        inline const Cell& operator()(const std::size_t i, const std::size_t j)
          const
        {
          return elements_[i + j * I_];
        }

        inline Cell& at(const std::size_t i, const std::size_t j)
        {
          return elements_.at(i + j * I_);
        }

        inline const Cell& at(const std::size_t i, const std::size_t j) const
        {
          return elements_.at(i + j * I_);
        }

        inline std::size_t to_global_index(
          const std::size_t i,
          const std::size_t j) const
        {
          return i + j * I_;
        }

        inline std::tuple<std::size_t, std::size_t> from_global_index(
          const std::size_t k)
        {
          return std::make_tuple((k - (k/I_) * I_), k / I_);
        }

      private:

        std::vector<Cell> elements_;

        std::size_t I_;
        std::size_t J_;
    };

    using Grid2d = Manifolds::Euclidean::PgmGeometry::Grid2d;

    using Grid2dMetricData =
      Manifolds::Euclidean::PgmGeometry::Grid2dMetricData;

    CellGrid() = delete;

    CellGrid(const Grid2dMetricData& grid_data, const Grid2d& grid);

    ~CellGrid() = default;

    Grid2dMetricData grid_data_;

    Grid grid_elements_;

    std::vector<std::shared_ptr<Cell>> fluid_cells_;
    std::vector<std::shared_ptr<Cell>> outlet_cells_;
    std::vector<std::shared_ptr<Cell>> inlet_cells_;
    std::vector<std::shared_ptr<Cell>> noslip_wall_cells_;
    std::vector<std::shared_ptr<Cell>> freeslip_wall_cells_;

  private:

    void assign_cell_types(const Grid2d& grid);

    void calculate_cell_wall_distances();
};

} // namespace PgmGeometry

} // namespace FiberBundles

#endif // FIBER_BUNDLES_PGM_GEOMETRY_CELL_GRID_H