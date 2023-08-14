#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/TurbulentFlow/ParseGeometryFile.h"
#include "gtest/gtest.h"

#include <cstddef> // std::size_t

using Utilities::FileIO::FilePath;
using Utilities::FileIO::TurbulentFlow::ParseGeometryFile;
using std::size_t;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace FileIO
{
namespace TurbulentFlow
{

static const std::string relative_turbulent_flow_configuration_path {
  "TurbulentCFDExampleCases"};

static const std::string relative_fluid_trap_path {"FluidTrap"};

static const std::string relative_step_flow_turb_path {
  "StepFlowTurb"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ParseGeometryFileTests, Constructible)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_step_flow_turb_path);
  fp.append("StepFlow.pgm");
  ParseGeometryFile read_pgm {fp};

  EXPECT_EQ(read_pgm.comments_.size(), 0);
  EXPECT_EQ(read_pgm.depth_, 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ParseGeometryFileTests, ParsesPgmFile)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_fluid_trap_path);
  fp.append("FluidTrap.pgm");
  ParseGeometryFile read_pgm {fp};

  read_pgm.parse_geometry_file();

  EXPECT_EQ(read_pgm.comments_.size(), 1);
  EXPECT_EQ(
    read_pgm.comments_.at(0),
    "# This is a test geometry file. filename: b_channel_box.pgm");
  EXPECT_EQ(read_pgm.depth_, 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ParseGeometryFileTests, ParsesPgmFileIntoGrid2d)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_fluid_trap_path);
    fp.append("FluidTrap.pgm");
    ParseGeometryFile read_pgm {fp};

    const auto grid = read_pgm.parse_geometry_file();

    EXPECT_EQ(grid.get_M(), 102-2);
    EXPECT_EQ(grid.get_N(), 52-2);

    for (size_t i {0}; i < grid.get_M() + 2; ++i)
    {
      EXPECT_EQ(grid.at(i, grid.get_N() + 1), 10);

      // Test values of the "second" and "third" column(s).
      if (i == 0)
      {
        EXPECT_EQ(grid.at(i, grid.get_N()), 11);
        EXPECT_EQ(grid.at(i, grid.get_N() - 1), 11);
 
        EXPECT_EQ(grid.at(i, 1), 11);
      }
      else if (i == 31 || i == 32)
      {
        EXPECT_EQ(grid.at(i, grid.get_N()), 10);
        EXPECT_EQ(grid.at(i, grid.get_N() - 1), 10);
      }
      else if (i == grid.get_M() + 1)
      {
        EXPECT_EQ(grid.at(i, grid.get_N()), 12);
        EXPECT_EQ(grid.at(i, grid.get_N() - 1), 12);

        EXPECT_EQ(grid.at(i, 1), 12);
      }
      else
      {
        EXPECT_EQ(grid.at(i, grid.get_N()), 0);
        EXPECT_EQ(grid.at(i, grid.get_N() - 1), 0);
      }

      if (i == 64 || i == 65)
      {
        EXPECT_EQ(grid.at(i, 1), 10);
      }

      EXPECT_EQ(grid.at(i, 0), 10);
    }
  }
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_step_flow_turb_path);
    fp.append("StepFlow.pgm");
    ParseGeometryFile read_pgm {fp};

    const auto grid = read_pgm.parse_geometry_file();

    EXPECT_EQ(read_pgm.comments_.size(), 1);
    EXPECT_EQ(
      read_pgm.comments_.at(0),
      "# Geometry for flow over a step");
    EXPECT_EQ(read_pgm.depth_, 10);

    EXPECT_EQ(grid.get_M(), 122-2);
    EXPECT_EQ(grid.get_N(), 62-2);

    for (size_t i {0}; i < grid.get_M() + 2; ++i)
    {
      EXPECT_EQ(grid.at(i, 0), 10);
      EXPECT_EQ(grid.at(i, grid.get_N() + 1), 10);

      for (size_t j {1}; j < grid.get_N() + 1; ++j)
      {
        if (j <= 30)
        {
          if (i < 21)
          {
            EXPECT_EQ(grid.at(i, j), 10);
          }
          else if (i == grid.get_M() + 1)
          {
            EXPECT_EQ(grid.at(i, j), 1);
          }
          else
          {
            EXPECT_EQ(grid.at(i, j), 0);
          }
        }
        else
        {
          if (i == 0)
          {
            EXPECT_EQ(grid.at(i, j), 2);
          }
          else if (i == grid.get_M() + 1)
          {
            EXPECT_EQ(grid.at(i, j), 1);
          }
          else
          {
            EXPECT_EQ(grid.at(i, j), 0);
          }
        }
      }
    }
  }
}

} // namespace TurbulentFlow
} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests