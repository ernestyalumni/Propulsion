#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_CALCULATE_TEMPERATURE_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_CALCULATE_TEMPERATURE_H

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SimpleHeat
{

struct DeviceParameters
{
  unsigned int width_;
  unsigned int height_;
  float speed_;
};

// Use extern to say this is defined elsewhere.
extern __constant__ DeviceParameters device_parameters;

__global__ void copy_constant_array(float* input, const float* constant_input);

__global__ void calculate_temperature(float* output, const float* input);

} // namespace SimpleHeat
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_CALCULATE_TEMPERATURE_H