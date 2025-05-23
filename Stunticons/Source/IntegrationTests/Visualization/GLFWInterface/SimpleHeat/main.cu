#include "IntegrationTests/Visualization/GLFWInterface/SimpleHeat/GeneralizedSimpleHeat.h"
#include "IntegrationTests/Visualization/GLFWInterface/SimpleHeat/SimpleHeat.h"
#include "Utilities/HandleUnsuccessfulCUDACall.h"
#include "Visualization/GLFWInterface/MakeAnimatedBitMap.h"

using IntegrationTests::Visualization::GLFWInterface::SimpleHeat::
  GeneralizedSimpleHeat;
using IntegrationTests::Visualization::GLFWInterface::SimpleHeat::SimpleHeat;
using IntegrationTests::Visualization::GLFWInterface::SimpleHeat::
  SimpleHeatWithLinearSaturation;
using IntegrationTests::Visualization::GLFWInterface::SimpleHeat::
  SimpleHeatWithSetSaturation;
using Visualization::GLFWInterface::create_make_animated_bit_map;

int main(int argc, char* argv[])
{
  constexpr std::size_t dimension {SimpleHeat::dimension_};

  auto make_animated_bit_map = create_make_animated_bit_map(
    dimension,
    dimension,
    "GPU simple heat");

  SimpleHeatWithSetSaturation simple_heat {};
  // You can also comment out the above line and uncomment out the following
  // line to run SimpleHeat with a different float to color function:
  // SimpleHeatWithLinearSaturation simple_heat {};

  make_animated_bit_map.run(simple_heat);
}