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
/// \param CompoundFPT Floating point type for the array of coefficients.
//------------------------------------------------------------------------------

// Template declaration for constant arrays
template<typename CompoundFPT>
extern __constant__ CompoundFPT cnu_coefficients_first_order[4];

template<typename CompoundFPT>
extern __constant__ CompoundFPT cnu_coefficients_second_order[4];

//------------------------------------------------------------------------------
/// \param hd_i The distance between grid points in ith direction
/// \param FPT The floating point type.
/// \param CompoundFPT Floating point type for the array of coefficients.
//------------------------------------------------------------------------------

template <typename FPT, typename CompoundFPT, int NU>
void auxiliary_set_first_order_coefficients(
  const FPT hd_i[2],
  const FPT unscaled_c_nus[NU],
  CompoundFPT* c_nus)
{
  for (int nu {0}; nu < NU; ++nu)
  {
    c_nus[nu].x = unscaled_c_nus[nu] * (static_cast<FPT>(1.0) / hd_i[0]);
    c_nus[nu].y = unscaled_c_nus[nu] * (static_cast<FPT>(1.0) / hd_i[1]);
  }

  Utilities::HandleUnsuccessfulCUDACall handle_memcpy_to_symbol {
    "Failed to copy c_nus to symbol on device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_to_symbol,
    cudaMemcpyToSymbol(
      cnu_coefficients_first_order<CompoundFPT>,
      c_nus,
      sizeof(CompoundFPT) * NU,
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

  auxiliary_set_first_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);

  delete[] c_nus;
}

template <typename FPT, typename CompoundFPT>
void set_first_order_coefficients_for_p2(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(2.0) / static_cast<FPT>(3.0),
    static_cast<FPT>(-1.0) / static_cast<FPT>(12.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_first_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);

  delete[] c_nus;
}

template <typename FPT, typename CompoundFPT>
void set_first_order_coefficients_for_p3(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(3.0) / static_cast<FPT>(4.0),
    static_cast<FPT>(-3.0) / static_cast<FPT>(20.0),
    static_cast<FPT>(1.0) / static_cast<FPT>(60.0),
    static_cast<FPT>(0.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_first_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);

  delete[] c_nus;
}

template <typename FPT, typename CompoundFPT>
void set_first_order_coefficients_for_p4(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(4.0) / static_cast<FPT>(5.0),
    static_cast<FPT>(-1.0) / static_cast<FPT>(5.0),
    static_cast<FPT>(4.0) / static_cast<FPT>(105.0),
    static_cast<FPT>(-1.0) / static_cast<FPT>(280.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_first_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);

  delete[] c_nus;
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds

#endif // MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_C_NU_COEFFICIENTS_H
