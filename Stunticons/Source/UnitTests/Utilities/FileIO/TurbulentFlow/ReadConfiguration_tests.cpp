#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/TurbulentFlow/ReadConfiguration.h"
#include "gtest/gtest.h"

#include <optional>
#include <string>

using Utilities::FileIO::FilePath;
using Utilities::FileIO::TurbulentFlow::ReadConfiguration;

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

static const std::string relative_lid_driven_cavity_path {
  "LidDrivenCavity"};

static const std::string relative_step_flow_turb_path {
  "StepFlowTurb"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadConfigurationTests, Constructible)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_lid_driven_cavity_path);
  fp.append("LidDrivenCavity.dat");
  ReadConfiguration read_tf {fp};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadConfigurationTests, Destructible)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_lid_driven_cavity_path);
    fp.append("LidDrivenCavity.dat");
    ReadConfiguration read_tf {fp};

    SUCCEED();
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadConfigurationTests, ReadFileReadsIntoStruct)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_lid_driven_cavity_path);
  fp.append("LidDrivenCavity.dat");
  ReadConfiguration read_tf {fp};

  const auto configuration = read_tf.read_file();

  EXPECT_EQ(*configuration.std_size_t_parameters_.imax_, 100);
  EXPECT_EQ(*configuration.std_size_t_parameters_.jmax_, 100);
  EXPECT_EQ(*configuration.std_size_t_parameters_.itermax_, 10000);
  EXPECT_EQ(*configuration.std_size_t_parameters_.i_processes_, 1);
  EXPECT_EQ(*configuration.std_size_t_parameters_.j_processes_, 1);

  EXPECT_EQ(*configuration.int_type_parameters_.solver_, 1);
  EXPECT_EQ(*configuration.int_type_parameters_.model_, 0);
  EXPECT_EQ(*configuration.int_type_parameters_.simulation_, 2);
  EXPECT_EQ(*configuration.int_type_parameters_.refine_, 3);
  EXPECT_EQ(configuration.int_type_parameters_.preconditioner_, std::nullopt);

  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.xlength_, 16);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.ylength_, 16);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.dt_, 0.05);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.t_end_, 5);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.tau_, 0.5);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.dt_value_, 1);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.eps_, 0.001);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.omg_, 1.7);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.gamma_, 0.5);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.re_, 100.0);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.gx_, 0.0);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.gy_, 0.0);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.pi_, 0.0);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.ui_, 0.0);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.vi_, 0.0);
  EXPECT_EQ(configuration.double_type_parameters_.nu_, std::nullopt);
  EXPECT_EQ(configuration.double_type_parameters_.pr_, std::nullopt);
  EXPECT_EQ(configuration.double_type_parameters_.beta_, std::nullopt);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadConfigurationTests, UnorderedMapsRemainEmpty)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_lid_driven_cavity_path);
  fp.append("LidDrivenCavity.dat");
  ReadConfiguration read_tf {fp};

  const auto configuration = read_tf.read_file();

  EXPECT_TRUE(
    configuration.unordered_map_type_parameters_.wall_temperatures_.empty());
  EXPECT_TRUE(
    configuration.unordered_map_type_parameters_.wall_velocities_.empty());
  EXPECT_TRUE(configuration.unordered_map_type_parameters_.inlet_Us_.empty());
  EXPECT_TRUE(configuration.unordered_map_type_parameters_.inlet_Vs_.empty());
  EXPECT_TRUE(configuration.unordered_map_type_parameters_.inlet_Ts_.empty());
  EXPECT_TRUE(configuration.unordered_map_type_parameters_.inlet_Ks_.empty());
  EXPECT_TRUE(configuration.unordered_map_type_parameters_.inlet_eps_.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadConfigurationTests, ReadFileReadsIntoUnorderedMaps)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_step_flow_turb_path);
  fp.append("StepFlowTurb.dat");
  ReadConfiguration read_tf {fp};

  const auto configuration = read_tf.read_file();

  EXPECT_TRUE(
    configuration.unordered_map_type_parameters_.wall_temperatures_.empty());
  EXPECT_TRUE(
    configuration.unordered_map_type_parameters_.wall_velocities_.empty());
  EXPECT_FALSE(configuration.unordered_map_type_parameters_.inlet_Us_.empty());
  EXPECT_FALSE(configuration.unordered_map_type_parameters_.inlet_Vs_.empty());
  EXPECT_TRUE(configuration.unordered_map_type_parameters_.inlet_Ts_.empty());
  EXPECT_FALSE(configuration.unordered_map_type_parameters_.inlet_Ks_.empty());
  EXPECT_FALSE(configuration.unordered_map_type_parameters_.inlet_eps_.empty());

  EXPECT_EQ(configuration.unordered_map_type_parameters_.inlet_Us_.size(), 1);
  EXPECT_EQ(configuration.unordered_map_type_parameters_.inlet_Vs_.size(), 1);
  EXPECT_EQ(configuration.unordered_map_type_parameters_.inlet_Ks_.size(), 1);
  EXPECT_EQ(configuration.unordered_map_type_parameters_.inlet_eps_.size(), 1);

  EXPECT_DOUBLE_EQ(
    configuration.unordered_map_type_parameters_.inlet_Us_.at(2), 1.0);
  EXPECT_DOUBLE_EQ(
    configuration.unordered_map_type_parameters_.inlet_Vs_.at(2), 0.0);

  EXPECT_DOUBLE_EQ(
    configuration.unordered_map_type_parameters_.inlet_Ks_.at(2), 0.003);
  EXPECT_DOUBLE_EQ(
    configuration.unordered_map_type_parameters_.inlet_eps_.at(2), 0.0005);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadConfigurationTests, ParseGrid2dMetricDataOutputsToStruct)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_step_flow_turb_path);
    fp.append("StepFlowTurb.dat");
    ReadConfiguration read_tf {fp};

    const auto configuration = read_tf.read_file();

    const auto data = read_tf.parse_grid2d_metric_data(configuration);

    EXPECT_EQ(data.minimum_i_, 0);
    EXPECT_EQ(data.maximum_i_, 122);
    EXPECT_EQ(data.minimum_j_, 0);
    EXPECT_EQ(data.maximum_j_, 62);
    EXPECT_EQ(data.M_, 120);
    EXPECT_EQ(data.N_, 60);
    EXPECT_EQ(data.total_number_of_cells_, 7564);

    EXPECT_DOUBLE_EQ(data.x_length_, 12);
    EXPECT_DOUBLE_EQ(data.y_length_, 2);

    EXPECT_DOUBLE_EQ(data.dx_, 0.1);
    EXPECT_DOUBLE_EQ(data.dy_, 0.033333333333333333);
  }
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_lid_driven_cavity_path);
    fp.append("LidDrivenCavity.dat");
    ReadConfiguration read_tf {fp};

    const auto configuration = read_tf.read_file();

    const auto data = read_tf.parse_grid2d_metric_data(configuration);

    EXPECT_EQ(data.minimum_i_, 0);
    EXPECT_EQ(data.maximum_i_, 802);
    EXPECT_EQ(data.minimum_j_, 0);
    EXPECT_EQ(data.maximum_j_, 802);
    EXPECT_EQ(data.M_, 800);
    EXPECT_EQ(data.N_, 800);
    EXPECT_EQ(data.total_number_of_cells_, 643204);

    EXPECT_DOUBLE_EQ(data.x_length_, 16);
    EXPECT_DOUBLE_EQ(data.y_length_, 16);

    EXPECT_DOUBLE_EQ(data.dx_, 0.02);
    EXPECT_DOUBLE_EQ(data.dy_, 0.02);
  }
}

} // namespace TurbulentFlow
} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests