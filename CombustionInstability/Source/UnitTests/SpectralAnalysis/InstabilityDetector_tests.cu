#include "SpectralAnalysis/InstabilityDetector.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using CombustionInstability::SpectralAnalysis::DetectorParameters;
using CombustionInstability::SpectralAnalysis::DetectionResult;
using CombustionInstability::SpectralAnalysis::detect_instability_peaks_kernel;

namespace GoogleUnitTests
{
namespace CombustionInstability
{
namespace SpectralAnalysis
{

static constexpr int kBlockSize = 256;

template <typename T>
T* to_device_typed(const std::vector<T>& h)
{
  T* d;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

template <typename T>
std::vector<T> to_host_typed(const T* d, const int n)
{
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

class InstabilityDetectorTests : public ::testing::Test
{
  protected:
    // 1024-point FFT at 100 kHz → 513 bins, Δf = 97.7 Hz
    static constexpr int kN = 1024;
    static constexpr float kFs = 100000.0f;
    static constexpr int kNumBins = kN / 2 + 1;

    DetectorParameters make_params() const
    {
      DetectorParameters p;
      p.num_bins = kNumBins;
      p.sample_rate_hz = kFs;
      p.fft_length = kN;
      p.sigma_threshold = 3.0f;
      p.peak_search_half_width = 3;
      return p;
    }
};

// Injecting a spike at bin 60 (≈5860 Hz, 1T mode) with high noise baseline.
// Detector should flag instability.
TEST_F(InstabilityDetectorTests, DetectsInjectedPeak)
{
  // Flat noise floor at 1.0, spike at bin 60 with value 100.
  std::vector<float> h_psd(kNumBins, 1.0f);
  const int spike_bin = 60;
  h_psd[spike_bin] = 100.0f;

  const std::vector<int> h_mode_bins = {spike_bin};

  float* d_psd = to_device_typed(h_psd);
  int* d_mode_bins = to_device_typed(h_mode_bins);
  DetectionResult* d_results;
  cudaMalloc(&d_results, sizeof(DetectionResult));

  const DetectorParameters params = make_params();
  detect_instability_peaks_kernel<<<1, 1>>>(
    d_results, d_psd, d_mode_bins, 1, params);
  cudaDeviceSynchronize();

  const auto results = to_host_typed(d_results, 1);
  cudaFree(d_psd);
  cudaFree(d_mode_bins);
  cudaFree(d_results);

  EXPECT_TRUE(results[0].instability_detected)
    << "Expected instability detection for SNR > 3-sigma";
  EXPECT_NEAR(results[0].peak_psd, 100.0f, 1e-4f);
  EXPECT_GT(results[0].snr_sigma, params.sigma_threshold);
}

// Flat noise PSD with no injected peak: should not trigger detection.
TEST_F(InstabilityDetectorTests, NoFalsePositiveOnFlatNoise)
{
  // All bins = 1.0 (perfectly flat noise — zero variance → SNR undefined).
  // With zero std, snr_sigma = 0 < threshold.
  std::vector<float> h_psd(kNumBins, 1.0f);
  const std::vector<int> h_mode_bins = {60, 99};  // 1T ≈ 5860 Hz, 2T ≈ 9668 Hz

  float* d_psd = to_device_typed(h_psd);
  int* d_mode_bins = to_device_typed(h_mode_bins);
  DetectionResult* d_results;
  cudaMalloc(&d_results, 2 * sizeof(DetectionResult));

  const DetectorParameters params = make_params();
  detect_instability_peaks_kernel<<<1, 2>>>(
    d_results, d_psd, d_mode_bins, 2, params);
  cudaDeviceSynchronize();

  const auto results = to_host_typed(d_results, 2);
  cudaFree(d_psd);
  cudaFree(d_mode_bins);
  cudaFree(d_results);

  for (int i = 0; i < 2; ++i)
  {
    EXPECT_FALSE(results[i].instability_detected)
      << "False positive at mode index " << i;
  }
}

// frequency_to_bin / bin_to_frequency round-trip consistency.
TEST_F(InstabilityDetectorTests, FrequencyBinConversion)
{
  const DetectorParameters params = make_params();

  // 1T mode at ~5860 Hz
  const float f_1T = 5859.375f;  // exactly bin 60 at N=1024, fs=100kHz
  const int bin = params.frequency_to_bin(f_1T);
  EXPECT_EQ(bin, 60);

  const float f_recovered = params.bin_to_frequency(bin);
  EXPECT_NEAR(f_recovered, f_1T, params.sample_rate_hz / params.fft_length);
}

// Multiple simultaneous modes: each reported independently.
TEST_F(InstabilityDetectorTests, MultipleModes)
{
  std::vector<float> h_psd(kNumBins, 1.0f);
  // Inject 1T (bin 60) and 2T (bin 99) peaks.
  h_psd[60] = 200.0f;
  h_psd[99] = 50.0f;

  const std::vector<int> h_mode_bins = {60, 99};

  float* d_psd = to_device_typed(h_psd);
  int* d_mode_bins = to_device_typed(h_mode_bins);
  DetectionResult* d_results;
  cudaMalloc(&d_results, 2 * sizeof(DetectionResult));

  const DetectorParameters params = make_params();
  const dim3 block(2);
  detect_instability_peaks_kernel<<<1, block>>>(
    d_results, d_psd, d_mode_bins, 2, params);
  cudaDeviceSynchronize();

  const auto results = to_host_typed(d_results, 2);
  cudaFree(d_psd);
  cudaFree(d_mode_bins);
  cudaFree(d_results);

  EXPECT_TRUE(results[0].instability_detected) << "1T mode not detected";
  EXPECT_TRUE(results[1].instability_detected) << "2T mode not detected";
  EXPECT_GT(results[0].peak_psd, results[1].peak_psd)
    << "1T peak should be larger than 2T peak";
}

} // namespace SpectralAnalysis
} // namespace CombustionInstability
} // namespace GoogleUnitTests
