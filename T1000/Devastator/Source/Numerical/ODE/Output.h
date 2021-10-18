#ifndef NUMERICAL_ODE_OUTPUT_H
#define NUMERICAL_ODE_OUTPUT_H

#include <cstddef>
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

  // Results stored in the matrix y_save_[0..count_ - 1][0..n_var_ - 1]
  //std::vector<std::vector<double>> y_save_;

  void init(const std::size_t neqn, const double xlo, const double xhi);

  void save(const double x, std::vector<double>& y);
};

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_OUTPUT_H
