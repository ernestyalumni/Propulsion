#ifndef MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_C_NU_COEFFICIENTS_H
#define MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_C_NU_COEFFICIENTS_H

// TODO: Check if this include has size_t instead of std::size_t
#include <cuda_runtime.h> 

#include "Utilities/HandleUnsuccessfulCUDACall.h"

namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

//------------------------------------------------------------------------------
/// \param FPT The floating point type.
//------------------------------------------------------------------------------

template <typename FPT>
struct CnuCoefficientsFirstOrder
{
  static constexpr size_t Size {4};
  extern __constant__ FPT c_nus[Size];
}

template <typename FPT>
struct CnuCoefficientsSecondOrder
{
  static constexpr size_t Size {4};
  extern __constant__ FPT c_nus[Size];
}

//------------------------------------------------------------------------------
/// \param hd_i The distance between grid points in ith direction
/// \param CompoundFPT Floating point type for the array of coefficients.
//------------------------------------------------------------------------------

template <typename FPT, typename CompoundFPT>
void auxiliary_set_first_order_coefficients(
  const FPT hd_i[2],
  const FPT unscaled_c_nus[4],
  CompoundFPT& c_nus[4])
{
  for (int nu {0}; nu < Nu; ++nu)
  {
    c_nus[nu].x = unscaled_c_nus[nu] * (static_cast<FPT>(1.0) / hd_i[0]);
    c_nus[nu].y = unscaled_c_nus[nu] * (static_cast<FPT>(1.0) / hd_i[1]);
  }

  Utilities::HandleUnsuccessfulCUDACall handle_memcpy_to_symbol {
    "Failed to copy c_nus to symbol on device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_to_symbol,
    cudaMemcpyToSymbol(
      CnuCoefficientsFirstOrder<FPT>::c_nus,
      c_nus,
      sizeof(CompoundFPT) * Nu),
      0,
      cudaMemcpyHostToDevice));
}

template <typename FPT, typename CompoundFPT>
void set_first_order_coefficients_for_p1(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(1.0) / static_cast<FPT>(2.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_first_order_coefficients<FPT, CompoundFPT>(
    hd_i,
    unscaled_c_nus,
    c_nus);

  delete[] c_nus;
}

template <typename FPT, typename CompoundFPT>
void set_first_order_coefficients_for_p1(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(1.0) / static_cast<FPT>(2.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_first_order_coefficients<FPT, CompoundFPT>(
    hd_i,
    unscaled_c_nus,
    c_nus);

  delete[] c_nus;
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds

#endif // MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_C_NU_COEFFICIENTS_H
