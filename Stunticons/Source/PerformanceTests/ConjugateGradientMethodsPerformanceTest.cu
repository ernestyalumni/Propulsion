#include "Algebra/Solvers/BiconjugateGradientStabilized.h"
#include "Algebra/Solvers/ConjugateGradient.h"
#include "Algebra/Solvers/SetupBiconjugateGradientStabilized.h"
#include "Algebra/Solvers/SetupConjugateGradient.h"
#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/ReadMatrixMarketFile.h"

#include <chrono>
#include <cmath> // std::abs
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple> // std::get
#include <vector>

using BiconjugateGradientStabilized =
  Algebra::Solvers::BiconjugateGradientStabilized;
using ConjugateGradient = Algebra::Solvers::ConjugateGradient;
using SetupBiconjugateGradientStabilized =
  Algebra::Solvers::SetupBiconjugateGradientStabilized;
using SetupConjugateGradient = Algebra::Solvers::SetupConjugateGradient;

using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadColumnVectorMarketFile;
using Utilities::FileIO::ReadMatrixMarketFile;
using std::cout;
using std::get;
using std::size_t;
using std::vector;

int main()
{
  static const std::string relative_sparse_matrix_example_path_1 {
    "SuiteSparseMatrixCollection/1537_OptimizationProblem/c-18"};

  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_sparse_matrix_example_path_1);
    fp.append("c-18.mtx");
    ReadMatrixMarketFile read_mtx {fp};

    const auto host_csr = read_mtx.read_into_csr();

    fp.remove_filename();
    fp.append("c-18_b.mtx");
    ReadColumnVectorMarketFile read_mtxb {fp};
    const auto host_b = read_mtxb.read_file();

    SetupBiconjugateGradientStabilized setup {host_csr, host_b};

    BiconjugateGradientStabilized cg {
      setup.A_,
      setup.b_,
      setup.morphism_,
      setup.operations_,
      350'000};

    cg.create_default_initial_guess(setup.x_);

    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();

    const auto result = cg.solve(
      setup.x_,
      setup.Ax_,
      setup.r_,
      setup.p_,
      setup.s_);

    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();

     // Calculate the duration
    auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the elapsed time
    cout <<
      "The elapsed time for Biconjugate Gradient Stabilized on c-18: " <<
      elapsed.count() <<
      " milliseconds\n";

    cout << "Results for Biconjugate Gradient Stabilized on c-18: " <<
      get<0>(result) << ", number of iterations: " << get<1>(result) << "\n";

    vector<double> h_x_output (setup.M_, 0.0);
    vector<double> y (setup.M_, 0.0);

    setup.x_.copy_device_output_to_host(h_x_output);

    host_csr.multiply(h_x_output, y);

    size_t number_of_errors {0};
    double total_error {0.0};

    for (size_t i {0}; i < setup.M_; ++i)
    {
      const double error {std::abs(y.at(i) - host_b.at(i))};
      total_error += error;

      if (error > 0.01)
      {
        cout << "i: " << i << " computed: " << y.at(i) << ", expected: " <<
          host_b.at(i) << "\n";

        number_of_errors += 1;
      }
    }

    cout << "Number of errors: " << number_of_errors << ", total error: " <<
      total_error << "\n";
  }
  // Conjugate Gradient fails.
  /*
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_sparse_matrix_example_path_1);
    fp.append("c-18.mtx");
    ReadMatrixMarketFile read_mtx {fp};

    const auto host_csr = read_mtx.read_into_float_csr();

    fp.remove_filename();
    fp.append("c-18_b.mtx");
    ReadColumnVectorMarketFile read_mtxb {fp};
    const auto host_b = read_mtxb.read_file_as_float();

    SetupConjugateGradient setup {host_csr, host_b};

    ConjugateGradient cg {
      setup.A_,
      setup.b_,
      setup.morphism_,
      setup.operations_,
      1'600'000};

    cg.create_default_initial_guess(setup.x_);

    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();

    const auto result = cg.solve(
      setup.x_,
      setup.Ax_,
      setup.r_,
      setup.p_);

    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();

     // Calculate the duration
    auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the elapsed time
    cout <<
      "The elapsed time for Conjugate Gradient on c-18: " <<
      elapsed.count() <<
      " milliseconds\n";

    cout << "Results for Conjugate Gradient on c-18: " <<
      get<0>(result) << ", number of iterations: " << get<1>(result) << "\n";

    vector<float> h_x_output (setup.M_, 0.0);
    vector<float> y (setup.M_, 0.0);

    setup.x_.copy_device_output_to_host(h_x_output);

    host_csr.multiply(h_x_output, y);

    size_t number_of_errors {0};
    float total_error {0.0};

    for (size_t i {0}; i < setup.M_; ++i)
    {
      const float error {std::abs(y.at(i) - host_b.at(i))};
      total_error += error;

      if (error > 0.01)
      {
        cout << "i: " << i << " computed: " << y.at(i) << ", expected: " <<
          host_b.at(i) << "\n";

        number_of_errors += 1;
      }
    }

    cout << "Number of errors: " << number_of_errors << ", total error: " <<
      total_error << "\n";
  }
  */
}