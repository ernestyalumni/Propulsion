#include "Stencil.h"
#include <cuda_fp16.h> // For __half

namespace Utilities
{
namespace Testing
{
namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

template struct Stencil<float, 1>;
template struct Stencil<float, 2>;
template struct Stencil<float, 3>;
template struct Stencil<float, 4>;

template struct Stencil<double, 1>;
template struct Stencil<double, 2>;
template struct Stencil<double, 3>;
template struct Stencil<double, 4>;

template struct Stencil<__half, 1>;
template struct Stencil<__half, 2>;
template struct Stencil<__half, 3>;
template struct Stencil<__half, 4>;

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
} // namespace Testing
} // namespace Utilities