#ifndef NUMERICAL_ODE_RK_METHODS_INTEGRATION_INPUTS_H
#define NUMERICAL_ODE_RK_METHODS_INTEGRATION_INPUTS_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <typename ContainerT, typename Field = double>
class IntegrationInputs
{
  public:

    IntegrationInputs() = delete;

    IntegrationInputs(
      const ContainerT& y_0,
      const Field x_1,
      const Field x_2,
      const Field h_0 = static_cast<Field>(0)
      ):
      y_0_{y_0},
      x_1_{x_1},
      x_2_{x_2},
      h_0_{h_0}
    {
      // TODO: assert that h_0_ has same sign as x_2 - x_1.

      if (h_0 == static_cast<Field>(0))
      {
        h_0_ = std::max(
          (x_2 - x_1) / static_cast<Field>(10),
          std::numeric_limits<Field>::epsilon());
      }
    }

    virtual ~IntegrationInputs() = default;

    ContainerT y_0_;
    Field x_1_;
    Field x_2_;
    Field h_0_;
};

template <typename ContainerT, typename Field = double>
class IntegrationInputsForDenseOutput
{
  public:

    IntegrationInputsForDenseOutput() = delete;

    IntegrationInputsForDenseOutput(
      const ContainerT& y_0,
      const Field x_1,
      const Field x_2,
      const std::size_t number_of_steps = 100
      ):
      y_0_{y_0},
      x_(number_of_steps + 1),
      x_1_{x_1},
      x_2_{x_2}
    {
      const Field h {(x_2 - x_1) / static_cast<Field>(number_of_steps)};
      h_ = (std::signbit(h) ? -1.0 : 1.0) *
        std::max(std::abs(h), std::numeric_limits<Field>::epsilon());

      Field x {x_1_};
      std::transform(
        x_.begin(),
        x_.end(),
        x_.begin(),
        [&x, this](const Field)
          {
            const Field x_to_insert {x};
            x += this->h_;
            return x_to_insert;
          });
    }

    ContainerT y_0_;
    std::vector<Field> x_;
    Field x_1_;
    Field x_2_;
    Field h_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_INTEGRATION_INPUTS_H
