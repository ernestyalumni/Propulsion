#include "CellGrid.h"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>

using std::array;
using std::make_optional;
using std::make_shared;
using std::size_t;

namespace FiberBundles
{

namespace PgmGeometry
{

CellGrid::Grid::Grid(const size_t M, const size_t N):
  // Consider this reference for initialization over reserve:
  // https://stackoverflow.com/questions/8928547/vector-initialization-or-reserve
  elements_((M + 2) * (N + 2)),
  I_{M + 2},
  J_{N + 2}
{}

CellGrid::CellGrid(const Grid2dMetricData& grid_data, const Grid2d& grid):
  grid_data_{grid_data},
  grid_elements_{grid_data_.M_, grid_data_.N_},
  fluid_cells_{},
  outlet_cells_{},
  inlet_cells_{},
  noslip_wall_cells_{},
  freeslip_wall_cells_{}
{
  assign_cell_types(grid);
  calculate_cell_wall_distances();
}

void CellGrid::assign_cell_types(const Grid2d& grid)
{
  for (size_t j {grid_data_.minimum_j_}; j < grid_data_.maximum_j_; ++j)
  {
    for (size_t i {grid_data_.minimum_i_}; i < grid_data_.maximum_i_; ++i)
    {
      const int grid_value {grid.get(i, j)};

      if (grid_value == 0)
      {
        if (i == grid_data_.minimum_i_ ||
          i == grid_data_.maximum_i_ - 1 ||
          j == grid_data_.minimum_j_ ||
          j == grid_data_.maximum_j_ - 1)
        {
          continue;
        }

        grid_elements_(i, j) = Cell{i, j, Cell::CellType::Fluid};
        fluid_cells_.push_back(make_shared<Cell>(grid_elements_(i, j)));
      }
      else if (grid_value == 1)
      {
        grid_elements_(i, j) = Cell{i, j, Cell::CellType::Outlet};
        outlet_cells_.push_back(make_shared<Cell>(grid_elements_(i, j)));
      }
      else if (grid_value >= 2 && grid_value <= 9)
      {
        grid_elements_(i, j) =
          Cell{i, j, Cell::CellType::Inlet, make_optional(grid_value)};
        inlet_cells_.push_back(make_shared<Cell>(grid_elements_(i, j)));
      }
      else if (grid_value >= 10 && grid_value <= 19)
      {
        grid_elements_(i, j) =
          Cell{i, j, Cell::CellType::NoSlipWall, make_optional(grid_value)};
        noslip_wall_cells_.push_back(make_shared<Cell>(grid_elements_(i, j)));
      }
      else if (grid_value >= 20)
      {
        grid_elements_(i, j) =
          Cell{i, j, Cell::CellType::FreeSlipWall, make_optional(grid_value)};
        freeslip_wall_cells_.push_back(make_shared<Cell>(grid_elements_(i, j)));        
      }
    }
  }

  // Corner cell neighbor assignment.

  // Bottom-Left Corner.
  size_t i {0};
  size_t j {0};
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i, j + 1)),
    Cell::BorderPosition::Top);
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i + 1, j)),
    Cell::BorderPosition::Right);
  if (
    grid_elements_(i, j).get_neighbor(Cell::BorderPosition::Top)->get_type() ==
      Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Top);
  }
  if (
    grid_elements_(i, j).get_neighbor(Cell::BorderPosition::Right)->get_type() ==
      Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Right);
  }

  // Top-Left Corner
  j = grid_data_.N_ + 1;
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i, j - 1)),
    Cell::BorderPosition::Bottom);
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i + 1, j)),
    Cell::BorderPosition::Right);
  if (
    grid_elements_(i, j).get_neighbor(
      Cell::BorderPosition::Bottom)->get_type() == Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Bottom);
  }
  if (
    grid_elements_(i, j).get_neighbor(Cell::BorderPosition::Right)->get_type() ==
      Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Right);
  }

  // Top-Right Corner
  i = grid_data_.M_ + 1;
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i, j - 1)),
    Cell::BorderPosition::Bottom);
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i - 1, j)),
    Cell::BorderPosition::Left);
  if (
    grid_elements_(i, j).get_neighbor(
      Cell::BorderPosition::Bottom)->get_type() == Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Bottom);
  }
  if (
    grid_elements_(i, j).get_neighbor(Cell::BorderPosition::Left)->get_type() ==
      Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Left);
  }

  // Bottom-Right Corner
  j = 0;
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i, j + 1)),
    Cell::BorderPosition::Top);
  grid_elements_(i, j).set_neighbor(
    make_shared<Cell>(grid_elements_(i - 1, j)),
    Cell::BorderPosition::Left);
  if (
    grid_elements_(i, j).get_neighbor(
      Cell::BorderPosition::Top)->get_type() == Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Top);
  }
  if (
    grid_elements_(i, j).get_neighbor(Cell::BorderPosition::Left)->get_type() ==
      Cell::CellType::Fluid)
  {
    grid_elements_(i, j).add_border(Cell::BorderPosition::Left);
  }

  // Bottom Cells
  for (size_t i {1}; i < grid_data_.M_ + 1; ++i)
  {
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i + 1, j)),
      Cell::BorderPosition::Right);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i - 1, j)),
      Cell::BorderPosition::Left);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i, j + 1)),
      Cell::BorderPosition::Top);
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Right)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Right);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Left)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Left);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Top)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Top);
    }
  }

  // Top Cells
  j = grid_data_.N_ + 1;
  for (size_t i {1}; i < grid_data_.M_ + 1; ++i)
  {
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i + 1, j)),
      Cell::BorderPosition::Right);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i - 1, j)),
      Cell::BorderPosition::Left);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i, j - 1)),
      Cell::BorderPosition::Bottom);
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Right)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Right);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Left)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Left);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Bottom)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Bottom);
    }
  }

  // Left Cells
  i = 0;
  for (size_t j {1}; j < grid_data_.N_ + 1; ++j)
  {
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i + 1, j)),
      Cell::BorderPosition::Right);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i, j - 1)),
      Cell::BorderPosition::Bottom);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i, j + 1)),
      Cell::BorderPosition::Top);
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Right)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Right);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Bottom)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Bottom);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Top)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Top);
    }
  }

  // Right Cells
  i = grid_data_.M_ + 1;
  for (size_t j {1}; j < grid_data_.N_ + 1; ++j)
  {
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i - 1, j)),
      Cell::BorderPosition::Left);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i, j - 1)),
      Cell::BorderPosition::Bottom);
    grid_elements_(i, j).set_neighbor(
      make_shared<Cell>(grid_elements_(i, j + 1)),
      Cell::BorderPosition::Top);
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Left)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Left);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Bottom)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Bottom);
    }
    if (
      grid_elements_(i, j).get_neighbor(
        Cell::BorderPosition::Top)->get_type() == Cell::CellType::Fluid)
    {
      grid_elements_(i, j).add_border(Cell::BorderPosition::Top);
    }
  }

  // Inner cells
  for (size_t j {1}; j < grid_data_.N_ + 1; ++j)
  {
    for (size_t i {1}; i < grid_data_.M_ + 1; ++i)
    {
      grid_elements_(i, j).set_neighbor(
        make_shared<Cell>(grid_elements_(i - 1, j)),
        Cell::BorderPosition::Left);
      grid_elements_(i, j).set_neighbor(
        make_shared<Cell>(grid_elements_(i + 1, j)),
        Cell::BorderPosition::Right);
      grid_elements_(i, j).set_neighbor(
        make_shared<Cell>(grid_elements_(i, j + 1)),
        Cell::BorderPosition::Top);
      grid_elements_(i, j).set_neighbor(
        make_shared<Cell>(grid_elements_(i, j - 1)),
        Cell::BorderPosition::Bottom);

      if (grid_elements_(i, j).get_type() != Cell::CellType::Fluid)
      {
        if (
          grid_elements_(i, j).get_neighbor(
            Cell::BorderPosition::Left)->get_type() == Cell::CellType::Fluid)
        {
          grid_elements_(i, j).add_border(Cell::BorderPosition::Left);
        }
        if (
          grid_elements_(i, j).get_neighbor(
            Cell::BorderPosition::Right)->get_type() == Cell::CellType::Fluid)
        {
          grid_elements_(i, j).add_border(Cell::BorderPosition::Right);
        }
        if (
          grid_elements_(i, j).get_neighbor(
            Cell::BorderPosition::Top)->get_type() == Cell::CellType::Fluid)
        {
          grid_elements_(i, j).add_border(Cell::BorderPosition::Top);
        }
        if (
          grid_elements_(i, j).get_neighbor(
            Cell::BorderPosition::Bottom)->get_type() == Cell::CellType::Fluid)
        {
          grid_elements_(i, j).add_border(Cell::BorderPosition::Bottom);
        }
      }
    }
  }
}

void CellGrid::calculate_cell_wall_distances()
{
  auto get_global_index = [this](const size_t i, const size_t j)
  {
    return i + grid_data_.maximum_i_ * j;
  };

  for (auto& cell_ptr : fluid_cells_)
  {
    const size_t i {cell_ptr->i()};    
    const size_t j {cell_ptr->j()};
    Cell::CellType cell_type {Cell::CellType::Default};
    // Distance going in the +x direction
    double distance_x_p {0};
    // Distance going in the +y direction
    double distance_y_p {0};
    // Distance going in the -x direction
    double distance_x_n {0};
    // Distance going in the -y direction
    double distance_y_n {0};

    // do while loop is an exit control loop, i.e. it checks the condition after
    // the body of the loop has been executed (body in do while loop will always
    // be executed at least once).
    // https://stackoverflow.com/questions/25233132/the-difference-between-while-and-do-while-c

    size_t index_i_p {i};
    do
    {
      index_i_p++;
      cell_type = grid_elements_(index_i_p, j).get_type();
      distance_x_p += grid_data_.dx_;
    }
    while ((cell_type == Cell::CellType::Fluid) &&
      (index_i_p < grid_data_.maximum_i_ - 1));

    if ((cell_type != Cell::CellType::FreeSlipWall) &&
      (cell_type != Cell::CellType::NoSlipWall))
    {
      distance_x_p = std::numeric_limits<double>::max();
    }

    size_t index_i_n {i};
    do
    {
      index_i_n--;
      cell_type = grid_elements_(index_i_n, j).get_type();
      distance_x_n += grid_data_.dx_;
    }
    while ((cell_type == Cell::CellType::Fluid) &&
      (index_i_n > 0));

    if ((cell_type != Cell::CellType::FreeSlipWall) &&
      (cell_type != Cell::CellType::NoSlipWall))
    {
      distance_x_n = std::numeric_limits<double>::max();
    }

    size_t index_j_p {j};
    do
    {
      index_j_p++;
      cell_type = grid_elements_(i, index_j_p).get_type();
      distance_y_p += grid_data_.dy_;
    }
    while ((cell_type == Cell::CellType::Fluid) &&
      (index_j_p < grid_data_.maximum_j_ - 1));

    if ((cell_type != Cell::CellType::FreeSlipWall) &&
      (cell_type != Cell::CellType::NoSlipWall))
    {
      distance_y_p = std::numeric_limits<double>::max();
    }

    size_t index_j_n {j};
    do
    {
      index_j_n--;
      cell_type = grid_elements_(i, index_j_n).get_type();
      distance_y_n += grid_data_.dy_;
    }
    while ((cell_type == Cell::CellType::Fluid) &&
      (index_j_n > 0));

    if ((cell_type != Cell::CellType::FreeSlipWall) &&
      (cell_type != Cell::CellType::NoSlipWall))
    {
      distance_y_n = std::numeric_limits<double>::max();
    }

    array<size_t, 4> indices {
      get_global_index(index_i_p, j),
      get_global_index(index_i_n, j),
      get_global_index(i, index_j_p),
      get_global_index(i, index_j_n)};

    array<double, 4> distances {
      distance_x_p,
      distance_x_n,
      distance_y_p,
      distance_y_n};

    double* minimum_distance {
      std::min_element(distances.begin(), distances.end())};

    // This obtains the index of the minimum element of the array.
    const long int minimum_element_index {minimum_distance - distances.begin()};
    
    cell_ptr->closest_distance_.emplace(*minimum_distance);
    cell_ptr->closest_wall_index_.emplace(static_cast<size_t>(
      minimum_element_index));
  }
}

} // namespace PgmGeometry

} // namespace FiberBundles
