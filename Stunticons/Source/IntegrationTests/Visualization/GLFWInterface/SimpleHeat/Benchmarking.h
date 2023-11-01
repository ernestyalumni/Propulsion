#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_BENCHMARKING_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_BENCHMARKING_H

#include "DataStructures/Array.h"

#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SimpleHeat
{

class Benchmarking
{
  public:

    Benchmarking();

    cudaEvent_t start_;
    cudaEvent_t stop_;

    float total_time_;
    float frames_;

  protected:

    bool initialize_events();
};

} // namespace SimpleHeat
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_SIMPLE_HEAT_H