#ifndef NUMERICAL_ODE_RHS_VAN_H
#define NUMERICAL_ODE_RHS_VAN_H

#include <vector>

namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
/// \brief Van der Pol's equation function object
/// \details This implementation attempts to follow as closely as possible the
/// implementation presented in the book.
/// \ref 17.0.4 A Quick-Start Example, Numerical Recipes, pp. 905.
//------------------------------------------------------------------------------
class RhsVan
{
  public:

    RhsVan(const double eps):
      eps_{eps}
    {}

    void operator()(
      const double x,
      std::vector<double>& y,
      std::vector<double>& dydx);

  private:

    double eps_;
};

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RHS_VAN_H
