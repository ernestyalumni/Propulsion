#include "Output.h"

#include <cstddef>
#include <vector>

namespace Numerical
{
namespace ODE
{

Output::Output(): 
  k_max_{0},
  count_{0},
  dense_{false}
{}

Output::Output(const std::size_t n_save):
  k_max_{500},
  n_save_{n_save},
  count_{0},
  x_save_(k_max_)
{
  dense_ = n_save_ > 0 ? true : false;
}

void Output::init(const std::size_t neqn, const double xlo, const double xhi)
{
  n_var_ = neqn;

  // Originally k_max_ == -1
  if (k_max_ == 0)
  {
    return;
  }

  y_save_.reserve(k_max_);

  if (dense_)
  {
    x1_ = xlo;
    x2_ = xhi;
    x_out_ = x1_;
    dx_out_ = (x2_ - x1_) / n_save_;
  }
}

void Output::resize()
{
  // Unused
  //const std::size_t k_old {k_max_};
  k_max_ *= 2;
  
  // Originally the previous values had to be saved.
  //std::vector<double> temp_vec {x_save_};
  
  x_save_.resize(k_max_);

  // Originally, the previous values had to be copied over.
  //for (std::size_t k {0}; k < k_old; ++k)
  //{
  //x_save_[k] = temp_vec[k];
  //}

  y_save_.resize(k_max_);
}

void Output::save(const double x, std::vector<double>& y)
{
  if (k_max_ == 0)
  {
    return;
  }

  for (std::size_t i {0}; i < n_var_; ++i)
  {
    y_save_[count_][i] = y[i];
  }

  x_save_[count_++] = x;
}

} // namespace ODE
} // namespace Numerical