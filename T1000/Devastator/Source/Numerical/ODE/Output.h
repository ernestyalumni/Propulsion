#ifndef NUMERICAL_ODE_OUTPUT_H
#define NUMERICAL_ODE_OUTPUT_H

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace Numerical
{
namespace ODE
{

struct Output
{
  //----------------------------------------------------------------------------
  /// \brief Default constructor that gives no output.
  /// \details Suppresses all output.
  /// \ref 17.0.3 The Output Object, pp. 904, Numerical Recipes.
  //----------------------------------------------------------------------------
  Output();

  //----------------------------------------------------------------------------
  /// \brief Provides dense output at n_save equally spaced intervals.
  /// \details If nsave <= 0, output is saved only at the actual integration
  /// steps. Otherwise, output at values of x of your choosing.
  /// \ref 17.0.3 The Output Object, pp. 904, Numerical Recipes.
  //----------------------------------------------------------------------------
  Output(const std::size_t n_save);

  // Results stored in "matrix" y_save_[0..count - 1][0..n_var_ - 1].
  // Originally, pp. 904 Numerical Recipes called for
  // y_save_[0..n_var_ - 1][0..count - 1]
  std::vector<std::vector<double>> y_save_;

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

  // Results stored in the vector x_save_[0..count_ - 1] and
  std::vector<double> x_save_;

  //----------------------------------------------------------------------------
  /// \details Called by Odeint ctor, which passes neqn, the number of
  /// equations, xlo, the starting point of the integration, and xhi, the ending
  /// point.
  //----------------------------------------------------------------------------
  void init(const std::size_t neqn, const double xlo, const double xhi);

  //----------------------------------------------------------------------------
  /// \brief Resize storage arrays by a factor of 2, keeping saved data.
  //----------------------------------------------------------------------------
  void resize();

  //----------------------------------------------------------------------------
  /// \details Invokes dense_out function of stepper routine to produce output
  /// at xout. Normally called by out rather than directly. Assumes that xout is
  /// between xold and xold+h, where the stepper must keep track of xold, the
  /// location of the previous step, and x=xold+h, the current step.
  //----------------------------------------------------------------------------
  template <class Stepper>
  void save_dense(Stepper& s, const double x_out, const double h)
  {
    if (count_ == k_max_)
    {
      resize();
    }

    for (std::size_t i {0}; i < n_var_; ++i)
    {
      y_save_[count_][i] = s.dense_out(i, x_out, h);
    }

    x_save_[count_++] = x_out;
  }

  void save(const double x, std::vector<double>& y);

  template <class Stepper>
  void out(
    const int nstp,
    const double x,
    std::vector<double>& y,
    Stepper& s,
    const double h)
  {
    if (!dense_)
    {
      throw std::runtime_error("Dense output not set in Output!");
    }

    if (nstp == -1)
    {
      save(x, y);
      x_out_ += dx_out_;
    }
    else
    {
      while ((x - x_out_) * (x2_ - x1_) > 0.0)
      {
        save_dense(s, x_out_, h);
        x_out_ += dx_out_;
      }
    }
  }
};

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_OUTPUT_H
