#include "IntegrationTests/Visualization/GLFWInterface/SimpleHeat/Benchmarking.h"
#include "Utilities/HandleUnsuccessfulCUDACall.h"

#include <cuda_runtime.h>

using Utilities::HandleUnsuccessfulCUDACall;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SimpleHeat
{

Benchmarking::Benchmarking():
  start_{},
  stop_{},
  total_time_{0.f},
  frames_{0.f}
{
  initialize_events();
}

bool Benchmarking::initialize_events()
{
  HandleUnsuccessfulCUDACall handle_create_event {"Failed to create event"};
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g7c581e3613a2110ba4d4e7fd5c7da418
  // Creates an event object for current device.
  handle_create_event(cudaEventCreate(&start_));
  handle_create_event(cudaEventCreate(&stop_));

  return handle_create_event.is_cuda_success();
}

} // namespace SimpleHeat
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests