#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Matrices/GenerateCompressedSparseRowMatrix.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"
#include "Optimization/ConjugateGradient.h"
#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <vector>

using Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;
using Algebra::Modules::Matrices::SparseMatrices::DenseVector;
using Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix;
using Algebra::Modules::Matrices::SparseMatrices::generate_tridiagonal_matrix;
using Algebra::Modules::Morphisms::SparseMatrixMorphismOnDenseVector;
using Algebra::Modules::Vectors::Array;
using Algebra::Modules::Vectors::CuBLASVectorOperations;
using Optimization::ConjugateGradient;
using std::size_t;

namespace GoogleUnitTests
{
namespace Optimization
{

struct SetupConjugateGradientTests
{
  static constexpr size_t M_ {1048576};
  static constexpr size_t number_of_nonzero_elements {
    (M_ - 2) * 3 + 4};

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  ConjugateGradient cg_;
  Array r_;
  CuBLASVectorOperations operations_;

  SetupConjugateGradientTests(const float alpha=1.0f, const float beta=0.0f):
    h_csr_{M_, M_, number_of_nonzero_elements},
    A_{M_, M_, number_of_nonzero_elements},
    x_{M_},
    Ax_{M_},
    morphism_{alpha, beta},
    cg_{},
    r_{M_},
    operations_{}
  {
    generate_tridiagonal_matrix(h_csr_);
    A_.copy_host_input_to_device(h_csr_);
    cg_.create_default_initial_guess(x_);

    morphism_.buffer_size(A_, x_, Ax_);

    std::vector<float> h_r (M_, 1.0f);
    r_.copy_host_input_to_device(h_r);
  }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, ConstructibleWithDefaultValues)
{
  ConjugateGradient cg {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, CreateDefaultInitialGuessCreates0)
{
  ConjugateGradient cg {};

  DenseVector x {69};

  cg.create_default_initial_guess(x);

  std::array<float, 69> h_output {};

  x.copy_device_output_to_host(h_output);

  for (size_t i {0}; i < 69; ++i)
  {
    EXPECT_FLOAT_EQ(h_output.at(i), 0.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, InitialStepWorks)
{
  SetupConjugateGradientTests setup {};
  const auto result = setup.cg_.initial_step(
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.x_,
    setup.Ax_,
    setup.r_);

  EXPECT_TRUE(result.has_value());
  EXPECT_FLOAT_EQ(*result, 1048576.0f);
}

} // namespace Optimization
} // namespace GoogleUnitTests  
