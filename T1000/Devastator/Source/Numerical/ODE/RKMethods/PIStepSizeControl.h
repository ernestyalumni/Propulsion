#ifndef NUMERICAL_ODE_RK_METHODS_PI_STEP_SIZE_CONTROL_H
#define NUMERICAL_ODE_RK_METHODS_PI_STEP_SIZE_CONTROL_H

#include <algorithm>
#include <cassert>
#include <cmath>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <typename Field = double>
class PIStepSizeControl
{
  public:

    static constexpr Field minimum_error_ {static_cast<Field>(1.0e-4)};

    PIStepSizeControl():
      previous_error_{minimum_error}
      is_rejected_{false}
    {}

    bool get_is_rejected() const
    {
      return is_rejected_;
    }

    Field accept_computed_step(const Field error)
    {
      assert(error >= static_cast<Field>(0));

      if (error <= 1)
      {
        // Bookkeeping for next call.
        previous_error_ = std::max(error, minimum_error_);
        is_rejected_ = false;
        return true;
      }

      is_rejected_ = true;
      return false;
    }

  private:

    Field previous_error_;
    bool is_rejected_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_PI_STEP_SIZE_CONTROL_H
