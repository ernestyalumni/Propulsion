#include "PerformanceTests/Algebra/Solvers/SetupBiconjugateGradientStabilizedWithEigen.h"
#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/ReadMatrixMarketFile.h"

#include <chrono>
#include <cmath> // std::abs
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple> // std::get

using SetupBiconjugateGradientStabilizedWithEigen =
  PerformanceTests::Algebra::Solvers::SetupBiconjugateGradientStabilizedWithEigen;

using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadColumnVectorMarketFile;
using Utilities::FileIO::ReadMatrixMarketFile;
using std::cout;
using std::get;

int main()
{
  static const std::string relative_sparse_matrix_example_path_1 {
    "SuiteSparseMatrixCollection/1537_OptimizationProblem/c-18"};

  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_sparse_matrix_example_path_1);
    fp.append("c-18.mtx");
    ReadMatrixMarketFile read_mtx {fp};

    const auto host_csr = read_mtx.read_file_as_compressed_sparse_row();

    fp.remove_filename();
    fp.append("c-18_b.mtx");
    ReadColumnVectorMarketFile read_mtxb {fp};
    const auto host_b = read_mtxb.read_file();

    SetupBiconjugateGradientStabilizedWithEigen setup {
      read_mtx.number_of_rows_,
      read_mtx.number_of_columns_};

    setup.insert_into_A(get<0>(host_csr), get<1>(host_csr), get<2>(host_csr));

    setup.insert_into_b(host_b);

    setup.setup_solving();

    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();

    setup.solve();

    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();

     // Calculate the duration
    auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print the elapsed time
    cout <<
      "The elapsed time for Eigen's Biconjugate Gradient Stabilized on c-18: " <<
      elapsed.count() <<
      " milliseconds\n";

    const auto y = setup.A_ * setup.x_;

    cout << "Sanity check: rows, columns, size of y: " << y.rows() <<
      ", " << y.cols() << ", " << y.size() << "\n";

    size_t number_of_errors {0};
    double total_error {0.0};

    for (size_t i {0}; i < setup.x_.rows(); ++i)
    {
      const double error {std::abs(y[i] - setup.b_[i])};
      total_error += error;

      if (error > 0.01)
      {
        cout << "i: " << i << " computed: " << y[i] << ", expected: " <<
          setup.b_[i] << "\n";

        number_of_errors += 1;
      }
    }

    cout << "Number of errors: " << number_of_errors << ", total error: " <<
      total_error << "\n";
  }
}