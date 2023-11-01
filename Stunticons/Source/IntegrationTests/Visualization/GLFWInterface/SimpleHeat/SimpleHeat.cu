#include "IntegrationTests/Visualization/GLFWInterface/SimpleHeat/SimpleHeat.h"

#include "IntegrationTests/Visualization/GLFWInterface/SimpleHeat/calculate_temperature.h"
#include "Utilities/HandleUnsuccessfulCudaCall.h"
#include "Visualization/float_to_color.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>

using Utilities::HandleUnsuccessfulCUDACall;
using Visualization::ColorConversion::float_to_color_with_set_saturation;

using std::size_t;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SimpleHeat
{

SimpleHeat::SimpleHeat():
  output_graphics_{get_image_size()},
  input_{get_image_size()},
  output_{get_image_size()},
  constant_input_{get_image_size()},
  benchmarks_{}
{
  if (!create_initial_conditions())
  {
    throw std::runtime_error("Failed to copy initial values to device arrays");
  }

  DeviceParameters parameters {dimension_, dimension_, speed_};

  HandleUnsuccessfulCUDACall handle_constant_copy {
    "Failed to copy parameters to constant memory"};

  // Copy over constant memory.
  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_constant_copy,
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9bcf02b53644eee2bef9983d807084c7
    // __host__ cudaError_t cudaMemcpyToSymbol(const void* symbol, const void*
    // src, size_t count, size_t offset = 0, cudaMemcpyKind kind =
    //  cudaMemcpyHostToDevice)
    cudaMemcpyToSymbol(
      device_parameters,
      reinterpret_cast<void*>(&parameters),
      sizeof(DeviceParameters),
      // offset - offset from start of symbol in bytes.
      0,
      cudaMemcpyHostToDevice));
}

bool SimpleHeat::create_initial_conditions()
{
  float* temp {new float[get_image_size()]};

  for (size_t i {0}; i < dimension_ * dimension_; ++i)
  {
    temp[i] = 0;
    const size_t x {i % dimension_};
    const size_t y {i / dimension_};

    // Initial conditions
    if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
    {
      temp[i] = max_temperature_;
    }
  }

  // More initial conditions
  temp[dimension_ * 100 + 100] = (max_temperature_ + min_temperature_) / 2;
  temp[dimension_ * 700 + 100] = min_temperature_;
  temp[dimension_ * 300 + 300] = min_temperature_;
  temp[dimension_ * 200 + 700] = min_temperature_;
  for (size_t y {800}; y < 900; ++y)
  {
    for (size_t x {400}; x < 500; ++x)
    {
      temp[x + y * dimension_] = min_temperature_;
    }
  }

  if (!constant_input_.copy_host_input_to_device(temp, get_image_size()))
  {
    delete [] temp;
    return false;
  }

  for (size_t y {800}; y < dimension_; ++y)
  {
    for (size_t x {0}; x < 200; ++x)
    {
      temp[x + y * dimension_] = max_temperature_;
    }
  }

  if (!input_.copy_host_input_to_device(temp, get_image_size()))
  {
    delete [] temp;
    return false;
  }

  delete [] temp;

  return true;
}

void SimpleHeat::run(uchar4* ptr, int)
{
  HandleUnsuccessfulCUDACall handle_event_record {"Failed to record event"};

  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446
  // cudaError_t cudaEventRecord(cudaEvent_T event, cudaStream_t stream = 0)
  // Records an event.
  // stream - Stream in which to record event.

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_event_record,
    cudaEventRecord(benchmarks_.start_, 0));

  const dim3 blocks_per_grid {dimension_ / threads_, dimension_ / threads_};

  const dim3 threads_per_block {threads_, threads_};

  for (size_t i {0}; i < time_steps_per_frame_; ++i)
  {
    copy_constant_array<<<blocks_per_grid, threads_per_block>>>(
      input_.elements_,
      constant_input_.elements_);

    calculate_temperature<<<blocks_per_grid, threads_per_block>>>(
      output_.elements_,
      input_.elements_);

    // Swap pointers.
    float* temp {input_.elements_};
    input_.elements_ = output_.elements_;
    output_.elements_ = temp;
  }

  float_to_color_with_set_saturation<<<blocks_per_grid, threads_per_block>>>(
    ptr,
    input_.elements_);

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_event_record,
    cudaEventRecord(benchmarks_.stop_, 0));

  HandleUnsuccessfulCUDACall handle_event_synchronization {
    "Failed to synchronize event"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_event_synchronization,
    // cudaError_t cudaEventSynchronize(cudaEvent_ event)
    // Waits for an event to complete.
    cudaEventSynchronize(benchmarks_.stop_));

  float elasped_time {0.f};

  HandleUnsuccessfulCUDACall handle_event_elapsed_time {
    "Failed to get elapsed time"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_event_elapsed_time,
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6
    // Computes the elapsed time between events (in milliseconds with resolution
    // around 0.5 microseconds).
    cudaEventElapsedTime(&elasped_time, benchmarks_.start_, benchmarks_.stop_));

  benchmarks_.total_time_ += elasped_time;
  ++benchmarks_.frames_;

  std::cout << "Average Time per frame: " << (benchmarks_.total_time_ /
    benchmarks_.frames_) << " ms\n";
}

} // namespace SimpleHeat
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests