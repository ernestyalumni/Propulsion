#include "SpectralAnalysis/PowerSpectralDensity.h"
#include "gtest/gtest.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using CombustionInstability::SpectralAnalysis::apply_hann_window_kernel;
using CombustionInstability::SpectralAnalysis::compute_psd_kernel;
using CombustionInstability::SpectralAnalysis::accumulate_psd_kernel;
using CombustionInstability::SpectralAnalysis::kHannWindowPower;

namespace GoogleUnitTests
{
namespace CombustionInstability
{
namespace SpectralAnalysis
{

static constexpr int kBlockSize = 256;

// Helper: allocate device buffer, copy from host, return device pointer.
float* to_device(const std::vector<float>& h)
{
  float* d;
  cudaMalloc(&d, h.size() * sizeof(float));
  cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}

// Helper: copy device buffer to host vector.
std::vector<float> to_host(const float* d, const int n)
{
  std::vector<float> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
  return h;
}

// Hann window should produce w(0) = w(N-1) = 0 and w(N/2) = 1.
TEST(HannWindowKernelTests, WindowShape)
{
  constexpr int N = 1024;
  std::vector<float> ones(N, 1.0f);
  float* d_signal = to_device(ones);

  const dim3 block(kBlockSize);
  const dim3 grid((N + kBlockSize - 1) / kBlockSize);
  apply_hann_window_kernel<<<grid, block>>>(d_signal, N);
  cudaDeviceSynchronize();

  const auto result = to_host(d_signal, N);
  cudaFree(d_signal);

  // w(0) = 0.5*(1 - cos(0)) = 0
  EXPECT_NEAR(result[0], 0.0f, 1e-6f);
  // w(N/2) = 0.5*(1 - cos(pi)) = 1
  EXPECT_NEAR(result[N / 2], 1.0f, 1e-5f);
  // w(N-1) ≈ 0 (cos(2pi*(N-1)/N) ≈ cos(2pi) = 1)
  EXPECT_NEAR(result[N - 1], 0.0f, 1e-3f);
}

// PSD of a pure sinusoid should show a peak at the correct frequency bin.
TEST(PowerSpectralDensityTests, PureSinusoidPeak)
{
  constexpr int N = 1024;
  constexpr float sample_rate = 100000.0f;  // 100 kHz
  constexpr float signal_freq = 5859.375f;  // exactly bin 60 at N=1024, fs=100kHz
  const int expected_bin = static_cast<int>(
    signal_freq * N / sample_rate + 0.5f);  // = 60

  // Generate pure sine wave.
  std::vector<float> h_signal(N);
  for (int n = 0; n < N; ++n)
  {
    h_signal[n] = std::sin(2.0f * M_PI * signal_freq * n / sample_rate);
  }

  float* d_signal = to_device(h_signal);

  // Apply Hann window.
  const dim3 block(kBlockSize);
  const dim3 grid((N + kBlockSize - 1) / kBlockSize);
  apply_hann_window_kernel<<<grid, block>>>(d_signal, N);
  cudaDeviceSynchronize();

  // Forward R2C FFT.
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_R2C, 1);

  cufftComplex* d_fft_out;
  const int num_bins = N / 2 + 1;
  cudaMalloc(&d_fft_out, num_bins * sizeof(cufftComplex));

  cufftExecR2C(plan, d_signal, d_fft_out);
  cudaDeviceSynchronize();

  // Compute PSD.
  float* d_psd;
  cudaMalloc(&d_psd, num_bins * sizeof(float));
  const dim3 grid_psd((num_bins + kBlockSize - 1) / kBlockSize);
  compute_psd_kernel<<<grid_psd, block>>>(
    d_psd, d_fft_out, N, kHannWindowPower * N);
  cudaDeviceSynchronize();

  const auto psd = to_host(d_psd, num_bins);

  cufftDestroy(plan);
  cudaFree(d_signal);
  cudaFree(d_fft_out);
  cudaFree(d_psd);

  // Peak should be at or near expected_bin.
  int peak_bin = 0;
  for (int k = 1; k < num_bins; ++k)
  {
    if (psd[k] > psd[peak_bin]) { peak_bin = k; }
  }

  EXPECT_EQ(peak_bin, expected_bin)
    << "Peak at bin " << peak_bin
    << " (" << (peak_bin * sample_rate / N) << " Hz)"
    << ", expected bin " << expected_bin
    << " (" << signal_freq << " Hz)";

  // Peak should be much larger than neighbouring bins (narrow peak for pure tone).
  const float peak_val = psd[peak_bin];
  const float neighbour_val = std::max(psd[peak_bin - 2], psd[peak_bin + 2]);
  EXPECT_GT(peak_val, neighbour_val * 10.0f)
    << "Peak is not prominent: peak=" << peak_val
    << " neighbour=" << neighbour_val;
}

// Accumulate two identical PSD frames: result should equal the input PSD.
TEST(AccumulatePSDTests, AverageOfTwoIdenticalFrames)
{
  constexpr int num_bins = 513;
  const std::vector<float> psd_frame(num_bins, 2.5f);

  float* d_running = to_device(std::vector<float>(num_bins, 0.0f));
  float* d_new = to_device(psd_frame);

  const dim3 block(kBlockSize);
  const dim3 grid((num_bins + kBlockSize - 1) / kBlockSize);

  accumulate_psd_kernel<<<grid, block>>>(d_running, d_new, num_bins, 1);
  cudaDeviceSynchronize();
  accumulate_psd_kernel<<<grid, block>>>(d_running, d_new, num_bins, 2);
  cudaDeviceSynchronize();

  const auto result = to_host(d_running, num_bins);
  cudaFree(d_running);
  cudaFree(d_new);

  for (int k = 0; k < num_bins; ++k)
  {
    EXPECT_NEAR(result[k], 2.5f, 1e-5f)
      << "Bin " << k << " mismatch";
  }
}

} // namespace SpectralAnalysis
} // namespace CombustionInstability
} // namespace GoogleUnitTests
