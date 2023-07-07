#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/TurbulentFlow/ReadTurbulentFlowConfiguration.h"
#include "Utilities/FileIO/TurbulentFlow/TurbulentFlowConfiguration.h"
#include "gtest/gtest.h"

#include <optional>

using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadTurbulentFlowConfiguration;
using Utilities::FileIO::TurbulentFlowConfiguration::StdSizeTParameters;
using Utilities::FileIO::TurbulentFlowConfiguration::
  create_std_size_t_parameters_map;
using std::nullopt;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace FileIO
{
namespace TurbulentFlowConfiguration
{

static const std::string relative_turbulent_flow_configuration_path {
  "TurbulentCFDExampleCases"};

static const std::string relative_lid_driven_cavity_path {
  "LidDrivenCavity"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StdSizeTParametersTests, Constructible)
{
  StdSizeTParameters parameters {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StdSizeTParametersTests, Destructible)
{
  {
    StdSizeTParameters parameters {};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StdSizeTParametersTests, CreateStdSizeTParametersMapCreatesMap)
{
  StdSizeTParameters parameters {};

  auto map = create_std_size_t_parameters_map(parameters);

  EXPECT_TRUE(map.find("imax") != map.end());
  EXPECT_TRUE(map.find("jmax") != map.end());
  EXPECT_TRUE(map.find("jproc") != map.end());
  EXPECT_TRUE(map.find("turbulent k-epsilon models") == map.end());
  EXPECT_TRUE(*(map.at("imax")) == nullopt);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StdSizeTParametersTests, CreateStdSizeTParametersMapCreatesMapThatMutates)
{
  StdSizeTParameters parameters {};

  auto map = create_std_size_t_parameters_map(parameters);

  *map.at("imax") = 42;
  *map.at("jmax") = 69;
  *map.at("itermax") = 420;
  *map.at("iproc") = 777;

  EXPECT_EQ(*parameters.imax_, 42);
  EXPECT_EQ(*parameters.jmax_, 69);
  EXPECT_EQ(*parameters.itermax_, 420);
  EXPECT_EQ(*parameters.i_processes_, 777);
  EXPECT_EQ(parameters.j_processes_, nullopt);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DoubleTypeParametersTests, InitializeMutates)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_turbulent_flow_configuration_path);
  fp.append(relative_lid_driven_cavity_path);
  fp.append("LidDrivenCavity.dat");
  ReadTurbulentFlowConfiguration read_tf {fp};

  auto configuration = read_tf.read_file();

  EXPECT_EQ(configuration.double_type_parameters_.nu_, nullopt);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.re_, 100);
  EXPECT_EQ(configuration.double_type_parameters_.pr_, nullopt);
  EXPECT_EQ(configuration.double_type_parameters_.alpha_, nullopt);
  EXPECT_EQ(configuration.double_type_parameters_.beta_, nullopt);

  configuration.double_type_parameters_.initialize();

  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.nu_, 0.01);
  EXPECT_EQ(configuration.double_type_parameters_.pr_, nullopt);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.alpha_, 0.0);
  EXPECT_DOUBLE_EQ(*configuration.double_type_parameters_.beta_, 0.0);
}

} // namespace TurbulentFlowConfiguration
} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests