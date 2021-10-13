#ifndef NUMERICAL_ODE_OUTPUT_H
#define NUMERICAL_ODE_OUTPUT_H

#include <cstddef>
#include <vector>

namespace Numerical
{
namespace ODE
{

template <typename ContainerX, typename ContainerY>
struct Output
{
  Output(): 
    k_max_{},
    dense_{false},
    count_{0}
  {}

  Output(const std::size_t n_save):
    k_max_{500},
    n_save_{n_save},
    count_{0},
    x_save_(k_max_)
  {
    dense_ = n_save_ > 0 ? true : false;
  }

  // Results stored in the vector x_save_[0..count_ - 1] and
  ContainerX x_save_;

  // Results stored in "matrix" y_save_[0..count - 1][0..n_var_ - 1].
  // Originally, pp. 904 Numerical Recipes called for
  // y_save_[0..n_var_ - 1][0..count - 1]
  ContainerY y_save_;

  double x1_;
  double x2_;
  double x_out_;
  double dx_out_;

  // Current capacity of storage arrays.
  std::size_t k_max_;

  std::size_t n_var_;

  // Number of intervals to save at for dense output.
  std::size_t n_save_;

  // Number of values actually saved.
  std::size_t count_;

  // True if dense output requested.
  bool dense_;

  template <typename T>
  void save(const double x, T& y)
  {
    if (k_max_ == 0)
    {
      return;
    }

    if (count_ == k_max_)
  }


};

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_OUTPUT_H
