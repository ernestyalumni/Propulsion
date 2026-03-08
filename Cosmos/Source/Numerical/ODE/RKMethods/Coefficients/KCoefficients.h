#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_K_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_K_COEFFICIENTS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace Coefficients
{

//------------------------------------------------------------------------------
/// \details Choice of std::vector<Field> over std::array is because it's an
/// aggregate and an aggregate shouldn't allocate memory dynamically.
/// \ref https://stackoverflow.com/questions/39548254/does-stdarray-guarantee-allocation-on-the-stack-only
/// \ref https://stackoverflow.com/questions/4424579/stdvector-versus-stdarray-in-c
//------------------------------------------------------------------------------
template <std::size_t S, typename ContainerT>
class KCoefficients : public std::vector<ContainerT>
{
  public:

    KCoefficients():
      std::vector<ContainerT>(S)
    {}

    KCoefficients(const std::initializer_list<ContainerT>& k_coefficients):
      std::vector<ContainerT>{k_coefficients}
    {
      this->resize(S);
    }

    // Copy constructor
    KCoefficients(const KCoefficients&) = default;

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    /// cf. https://en.cppreference.com/w/cpp/language/copy_assignment
    //--------------------------------------------------------------------------
    KCoefficients& operator=(const KCoefficients&) = default;

    virtual ~KCoefficients() = default;

    const ContainerT& get_ith_coefficient(const std::size_t i) const
    {
      assert(i >= 1 && i <= S);

      return this->operator[](i - 1);
    }

    ContainerT& ith_coefficient(const std::size_t i)
    {
      return this->operator[](i - 1);
    }

    template <typename Field>
    void fill(const std::size_t i, const Field& value)
    {
      std::fill(
        this->operator[](i - 1).begin(),
        this->operator[](i - 1).end(),
        value);
    }

    template <
      typename OutT,
      typename Field,
      typename =
        typename std::enable_if<
          !std::is_same_v<OutT, std::vector<Field>>
          >::type
      >
    void scalar_multiply(
      OutT& out,
      const std::size_t k_coefficient_index,
      const Field scalar) const
    {
      std::transform(
        this->get_ith_coefficient(k_coefficient_index).begin(),
        this->get_ith_coefficient(k_coefficient_index).end(),
        out.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          scalar));
    }

    template <typename Field>
    void scalar_multiply(
      std::vector<Field>& out,
      const std::size_t k_coefficient_index,
      const Field scalar) const
    {
      assert(out.size() == S || out.empty());

      if (out.empty())
      {
        std::transform(
          this->get_ith_coefficient(k_coefficient_index).begin(),
          this->get_ith_coefficient(k_coefficient_index).end(),
          std::back_inserter(out),
          std::bind(
            std::multiplies<Field>(),
            std::placeholders::_1,
            scalar));
      }
      else
      {
        std::transform(
          this->get_ith_coefficient(k_coefficient_index).begin(),
          this->get_ith_coefficient(k_coefficient_index).end(),
          out.begin(),
          std::bind(
            std::multiplies<Field>(),
            std::placeholders::_1,
            scalar));
      }
    }

    template <typename Field>
    std::array<Field, S> scalar_multiply(
      const std::size_t k_coefficient_index,
      const Field scalar) const
    {
      std::array<Field, S> out {};
      std::transform(
        this->get_ith_coefficient(k_coefficient_index).begin(),
        this->get_ith_coefficient(k_coefficient_index).end(),
        out.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          scalar));

      return out;
    }

  protected:

    using std::vector<ContainerT>::emplace_back;
    using std::vector<ContainerT>::push_back;
    using std::vector<ContainerT>::pop_back;
};

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_K_COEFFICIENTS_H
