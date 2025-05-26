#ifndef MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_C_NU_COEFFICIENTS_H
#define MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_C_NU_COEFFICIENTS_H

// TODO: Check if this include has size_t instead of std::size_t
#include <cuda_runtime.h> 
#include <cuda_fp16.h>

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

//------------------------------------------------------------------------------
/// Be careful of usage. The recommended usage is that you do the template
/// specialization directly in the source files of your single object. For
/// example, we do / declare this:
/// 
/// template<>
/// __constant__ float2 cnu_coefficients_first_order<float2>[4];
///
/// In C_Nu_Coefficients_tests.cu (choosing this particular source file isn't
/// special, we could've placed the code in DirectionalDerivatives_tests.cu). We
/// do not declare it again in any other source file because of linkage error /
/// one definition rule (you'll obtain an error link the following):
///
/// /usr/bin/ld: CMakeFiles/Check.dir/Manifolds/Operators/FiniteDifference/DirectionalDerivatives_tests.cu.o:(.bss+0x0): multiple definition of `Manifolds::Operators::FiniteDifference::cnu_coefficients_first_order<float2>'; CMakeFiles/Check.dir/Manifolds/Operators/FiniteDifference/C_Nu_Coefficients_tests.cu.o:(.bss+0x0): first defined here
/// collect2: error: ld returned 1 exit status
/// make[2]: *** [UnitTests/CMakeFiles/Check.dir/build.make:818: Check] Error 1
/// make[1]: *** [CMakeFiles/Makefile2:898: UnitTests/CMakeFiles/Check.dir/all] Error 2
/// make: *** [Makefile:136: all] Error 2
///
/// For your future application, you should do the specialization directly in
/// the source file of your single object.
//------------------------------------------------------------------------------
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
    c_nus[nu].x = unscaled_c_nus[nu] * (static_cast<FPT>(1.) / hd_i[0]);
    c_nus[nu].y = unscaled_c_nus[nu] * (static_cast<FPT>(1.) / hd_i[1]);
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

template <typename FPT, typename CompoundFPT, int NU>
void auxiliary_set_second_order_coefficients(
  const FPT hd_i[2],
  const FPT unscaled_c_nus[NU],
  CompoundFPT* c_nus)
{
  for (int nu {0}; nu < NU; ++nu)
  {
    c_nus[nu].x = unscaled_c_nus[nu] * (
      static_cast<FPT>(1.0) /
        (hd_i[0] * hd_i[0]));
    c_nus[nu].y = unscaled_c_nus[nu] * (
      static_cast<FPT>(1.0) /
        (hd_i[1] * hd_i[1]));
  }

  Utilities::HandleUnsuccessfulCUDACall handle_memcpy_to_symbol {
    "Failed to copy c_nus to symbol on device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy_to_symbol,
    cudaMemcpyToSymbol(
      cnu_coefficients_second_order<CompoundFPT>,
      c_nus,
      sizeof(CompoundFPT) * NU,
      0,
      cudaMemcpyHostToDevice));
}

template <typename FPT, typename CompoundFPT>
void set_second_order_coefficients_for_p1(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(1.0) / static_cast<FPT>(2.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_second_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);
  
  delete[] c_nus;
}

template <typename FPT, typename CompoundFPT>
void set_second_order_coefficients_for_p2(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(4.0) / static_cast<FPT>(3.0),
    static_cast<FPT>(-1.0) / static_cast<FPT>(12.0),
    static_cast<FPT>(0.0),
    static_cast<FPT>(0.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_second_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);
  
  delete[] c_nus;
}

template <typename FPT, typename CompoundFPT>
void set_second_order_coefficients_for_p3(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(3.0) / static_cast<FPT>(2.0),
    static_cast<FPT>(-3.0) / static_cast<FPT>(20.0),
    static_cast<FPT>(1.0) / static_cast<FPT>(90.0),
    static_cast<FPT>(0.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_second_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);
  
  delete[] c_nus;
}

template <typename FPT, typename CompoundFPT>
void set_second_order_coefficients_for_p4(const FPT hd_i[2])
{
  static constexpr int Nu {4};
  static constexpr FPT unscaled_c_nus[Nu] {
    static_cast<FPT>(8.0) / static_cast<FPT>(5.0),
    static_cast<FPT>(-1.0) / static_cast<FPT>(5.0),
    static_cast<FPT>(8.0) / static_cast<FPT>(315.0),
    static_cast<FPT>(-1.0) / static_cast<FPT>(560.0)
  };

  CompoundFPT* c_nus {new CompoundFPT[Nu]};

  auxiliary_set_second_order_coefficients<FPT, CompoundFPT, Nu>(
    hd_i,
    unscaled_c_nus,
    c_nus);
  
  delete[] c_nus;
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds

#endif // MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_C_NU_COEFFICIENTS_H
