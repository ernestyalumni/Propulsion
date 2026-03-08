#ifndef NUMERICAL_ODE_RHS_VAN_DER_POL_H
#define NUMERICAL_ODE_RHS_VAN_DER_POL_H

namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
/// \brief Van der Pol's equation function object
/// \ref 17.0.4 A Quick-Start Example, Numerical Recipes, pp. 905.
//------------------------------------------------------------------------------
template <typename Container>
class RhsVanDerPol
{
  public:

    RhsVanDerPol(const double eps, const double mu = 1.0):
      eps_{eps},
      mu_{mu}
    {}

    void operator()(
      const double,
      Container& y,
      Container& dydx)
    {
      dydx[0] = y[1];
      dydx[1] = (mu_ * (1.0 - y[0] * y[0]) * y[1] - y[0]) / eps_;
    }

  private:

    double eps_;
    double mu_;
};

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RHS_VAN_DER_POL_H
