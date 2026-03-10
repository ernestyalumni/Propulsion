#ifndef ASTRODYNAMICS_SPECIFIC_ENERGY_H
#define ASTRODYNAMICS_SPECIFIC_ENERGY_H

#include "Algebra/Modules/Vectors/Vector3.h"

#include <concepts>

namespace Astrodynamics
{

//------------------------------------------------------------------------------
/// Orbital mechanics helper
//------------------------------------------------------------------------------
template <std::floating_point Field = double>
struct SpecificEnergy
{
  using Vector3 = Algebra::Modules::Vectors::Vector3<Field>;
  Field mu_;

  explicit SpecificEnergy(Field mu):
    mu_(mu)
  {}

  inline Field operator()(const Vector3& r, const Vector3& v) const
  {
    return v.norm_squared() / static_cast<Field>(2.) - mu_ / r.norm();
  }
};

} // namespace Astrodynamics

#endif // ASTRODYNAMICS_SPECIFIC_ENERGY_H