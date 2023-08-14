#include "FiberBundles/PgmGeometry/Cell.h"
#include "FiberBundles/PgmGeometry/CellGrid.h"
#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/TurbulentFlow/ParseGeometryFile.h"
#include "Utilities/FileIO/TurbulentFlow/ReadConfiguration.h"
#include "gtest/gtest.h"

#include <string>

using FiberBundles::PgmGeometry::Cell;
using FiberBundles::PgmGeometry::CellGrid;
using Utilities::FileIO::FilePath;
using Utilities::FileIO::TurbulentFlow::ParseGeometryFile;
using Utilities::FileIO::TurbulentFlow::ReadConfiguration;

static const std::string relative_turbulent_flow_configuration_path {
  "TurbulentCFDExampleCases"};

static const std::string relative_fluid_trap_path {"FluidTrap"};

static const std::string relative_step_flow_turb_path {
  "StepFlowTurb"};

namespace GoogleUnitTests
{
namespace FiberBundles
{
namespace PgmGeometry
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CellGridTests, Constructible)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_fluid_trap_path);
    fp.append("FluidTrap.dat");
    ReadConfiguration read_tf {fp};

    const auto configuration = read_tf.read_file();

    const auto data = read_tf.parse_grid2d_metric_data(configuration);

    FilePath fp2 {FilePath::get_data_directory()};
    fp2.append(relative_turbulent_flow_configuration_path);
    fp2.append(relative_fluid_trap_path);
    fp2.append("FluidTrap.pgm");
    ParseGeometryFile read_pgm {fp2};

    const auto grid = read_pgm.parse_geometry_file();

    CellGrid cg {data, grid};

    EXPECT_FALSE(cg.grid_elements_(0, 0).is_border(Cell::BorderPosition::Bottom));
    EXPECT_FALSE(cg.grid_elements_(0, 0).is_border(Cell::BorderPosition::Left));
    EXPECT_FALSE(cg.grid_elements_(0, 0).is_border(Cell::BorderPosition::Top));
    EXPECT_FALSE(cg.grid_elements_(0, 0).is_border(Cell::BorderPosition::Right));

    EXPECT_TRUE(cg.grid_elements_(0, 0).get_id().has_value());
    EXPECT_EQ(*(cg.grid_elements_(0, 0).get_id()), 10);

    EXPECT_EQ(
      cg.grid_elements_(0, 0).get_neighbor(
        Cell::BorderPosition::Top)->get_type(),
      Cell::CellType::NoSlipWall);
    EXPECT_EQ(
      cg.grid_elements_(0, 0).get_neighbor(
        Cell::BorderPosition::Right)->get_type(),
      Cell::CellType::NoSlipWall);
  }
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_step_flow_turb_path);
    fp.append("StepFlowTurb.dat");
    ReadConfiguration read_tf {fp};

    const auto configuration = read_tf.read_file();

    const auto data = read_tf.parse_grid2d_metric_data(configuration);

    FilePath fp2 {FilePath::get_data_directory()};
    fp2.append(relative_turbulent_flow_configuration_path);
    fp2.append(relative_step_flow_turb_path);
    fp2.append("StepFlow.pgm");
    ParseGeometryFile read_pgm {fp2};

    const auto grid = read_pgm.parse_geometry_file();

    CellGrid cg {data, grid};

    EXPECT_FALSE(cg.grid_elements_(1, 0).is_border(Cell::BorderPosition::Bottom));
    EXPECT_FALSE(cg.grid_elements_(1, 0).is_border(Cell::BorderPosition::Left));
    EXPECT_FALSE(cg.grid_elements_(1, 0).is_border(Cell::BorderPosition::Top));
    EXPECT_FALSE(cg.grid_elements_(1, 0).is_border(Cell::BorderPosition::Right));

    EXPECT_TRUE(cg.grid_elements_(1, 0).get_id().has_value());
    EXPECT_EQ(*(cg.grid_elements_(1, 0).get_id()), 10);

    EXPECT_EQ(
      cg.grid_elements_(1, 0).get_neighbor(
        Cell::BorderPosition::Top)->get_type(),
      Cell::CellType::NoSlipWall);
    EXPECT_EQ(
      cg.grid_elements_(1, 0).get_neighbor(
        Cell::BorderPosition::Left)->get_type(),
      Cell::CellType::NoSlipWall);
    EXPECT_EQ(
      cg.grid_elements_(1, 0).get_neighbor(
        Cell::BorderPosition::Right)->get_type(),
      Cell::CellType::NoSlipWall);

    EXPECT_FALSE(cg.grid_elements_(121, 1).is_border(Cell::BorderPosition::Bottom));
    EXPECT_TRUE(cg.grid_elements_(121, 1).is_border(Cell::BorderPosition::Left));
    EXPECT_FALSE(cg.grid_elements_(121, 1).is_border(Cell::BorderPosition::Top));
    EXPECT_FALSE(cg.grid_elements_(121, 1).is_border(Cell::BorderPosition::Right));

    EXPECT_FALSE(cg.grid_elements_(121, 1).get_id().has_value());

    EXPECT_EQ(
      cg.grid_elements_(121, 1).get_neighbor(
        Cell::BorderPosition::Top)->get_type(),
      Cell::CellType::Outlet);
    EXPECT_EQ(
      cg.grid_elements_(121, 1).get_neighbor(
        Cell::BorderPosition::Bottom)->get_type(),
      Cell::CellType::NoSlipWall);
    EXPECT_EQ(
      cg.grid_elements_(121, 1).get_neighbor(
        Cell::BorderPosition::Left)->get_type(),
      Cell::CellType::Fluid);
  }
}

} // namespace PgmGeometry
} // namespace FiberBundles

} // namespace GoogleUnitTests