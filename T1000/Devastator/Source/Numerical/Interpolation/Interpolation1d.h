//------------------------------------------------------------------------------
/// \ref 3.1 Preliminaries: Searching an Ordered Table, pp. 114, Numerical
/// Recipes, 3rd. Ed.
//------------------------------------------------------------------------------

struct Base_interp
//------------------------------------------------------------------------------
/// \brief Abstract base class used by all interpolation routines in this
///   chapter. Only routine interp called directly by user.
//------------------------------------------------------------------------------
{
  int n, mm, jsav, cor, dj;
  const double *xx, *yy;

  //----------------------------------------------------------------------------
  /// \brief Constructor: Set up for interpolating on a table of x's and y's of
  /// length m. Normally called by a derived class, not by the user.
  //----------------------------------------------------------------------------
  Base_interp(const double* y, int m):
    n(x.size()),
    mm(m),
    jsav(0),
    cor(0),
    xx(&x[0]),
    yy(y)
  {
    dj = std::min(1, (int)std::pow((double)n, 0.25));
  }



  //----------------------------------------------------------------------------
  /// \brief Derived classes provide this as the actual interpolation method.
  //----------------------------------------------------------------------------
  double virtual rawinterp(int jlo, double x) = 0;

};

