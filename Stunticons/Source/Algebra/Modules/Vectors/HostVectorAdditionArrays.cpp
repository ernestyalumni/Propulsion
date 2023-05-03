#include "HostVectorAdditionArrays.h"

#include <cstddef> // std::size_t
#include <cstdlib> // free

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

HostVectorAdditionArrays::HostVectorAdditionArrays(
  const std::size_t input_size
  ):
  number_of_elements_{input_size},
  h_A_{static_cast<float*>(malloc(input_size * sizeof(float)))},
  h_B_{static_cast<float*>(malloc(input_size * sizeof(float)))},
  h_C_{static_cast<float*>(malloc(input_size * sizeof(float)))}
{}

HostVectorAdditionArrays::~HostVectorAdditionArrays()
{
  free(h_A_);
  free(h_B_);
  free(h_C_);
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra