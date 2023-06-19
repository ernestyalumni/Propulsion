#include "Utilities/FileIO/ReadTurbulentFlowConfiguration.h"
#include "Utilities/FileIO/FilePath.h"
#include "gtest/gtest.h"

#include <optional>

using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadTurbulentFlowConfiguration;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace FileIO
{

static const std::string relative_turbulent_flow_configuration_path {
  "TurbulentCFDExampleCases"};

static const std::string relative_lid_driven_cavity_path {
  "LidDrivenCavity"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadTurbulentFlowConfigurationTests, Constructible)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_lid_driven_cavity_path);
  fp.append("LidDrivenCavity.dat");
  ReadTurbulentFlowConfiguration read_tf {fp};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadTurbulentFlowConfigurationTests, Destructible)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_turbulent_flow_configuration_path);
    fp.append(relative_lid_driven_cavity_path);
    fp.append("LidDrivenCavity.dat");
    ReadTurbulentFlowConfiguration read_tf {fp};

    SUCCEED();
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadTurbulentFlowConfigurationTests, ReadFileReadsIntoStruct)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_lid_driven_cavity_path);
  fp.append("LidDrivenCavity.dat");
  ReadTurbulentFlowConfiguration read_tf {fp};

  const auto configuration = read_tf.read_file();

  EXPECT_EQ(*configuration.std_size_t_parameters_.imax_, 100);
  EXPECT_EQ(*configuration.std_size_t_parameters_.jmax_, 100);
  EXPECT_EQ(*configuration.std_size_t_parameters_.itermax_, 10000);
  EXPECT_EQ(*configuration.std_size_t_parameters_.iproc_, 1);
  EXPECT_EQ(*configuration.std_size_t_parameters_.jproc_, 1);

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

} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests