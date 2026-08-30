/**
 * PSD pipeline benchmark: measures GPU throughput for the full
 * Hann-window → cuFFT → PSD → detection pipeline.
 *
 * Simulates a 100 kHz pressure sensor with an injected 1T instability mode
 * at 5859.375 Hz (bin 60 for N=1024, fs=100 kHz).
 *
 * Reports:
 *   - Time per FFT frame (µs)
 *   - Throughput (frames/s, samples/s)
 *   - Detection result for the injected mode
 *
 * Build: cmake ../Source && make BenchmarkCombustionInstability
 * Run:   ./BenchmarkCombustionInstability
 */

#include "SpectralAnalysis/PowerSpectralDensity.h"
#include "SpectralAnalysis/InstabilityDetector.h"

#include <cufft.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using CombustionInstability::SpectralAnalysis::apply_hann_window_kernel;
using CombustionInstability::SpectralAnalysis::compute_psd_kernel;
using CombustionInstability::SpectralAnalysis::accumulate_psd_kernel;
using CombustionInstability::SpectralAnalysis::detect_instability_peaks_kernel;
using CombustionInstability::SpectralAnalysis::DetectorParameters;
using CombustionInstability::SpectralAnalysis::DetectionResult;
using CombustionInstability::SpectralAnalysis::kHannWindowPower;

static constexpr int kNfft = 1024;
static constexpr float kFs = 100000.0f;  // 100 kHz
static constexpr float kF1T = 5859.375f; // exactly bin 60
static constexpr float kInstabilityAmplitude = 5.0f;
static constexpr float kNoiseAmplitude = 1.0f;
static constexpr int kNumFrames = 1000;
static constexpr int kBlockSize = 256;

int main()
{
  printf("=== Combustion Instability PSD Pipeline Benchmark ===\n");
  printf("N_fft=%d  fs=%.0f Hz  f_1T=%.2f Hz  frames=%d\n\n",
         kNfft, kFs, kF1T, kNumFrames);

  const int num_bins = kNfft / 2 + 1;

  // ── Synthesize signal: noise + 1T instability tone ──────────────────────
  std::vector<float> h_signal(kNfft);
  srand(42);
  for (int n = 0; n < kNfft; ++n)
  {
    const float noise = kNoiseAmplitude * (2.0f * rand() / RAND_MAX - 1.0f);
    const float tone = kInstabilityAmplitude *
      std::sin(2.0f * M_PI * kF1T * n / kFs);
    h_signal[n] = noise + tone;
  }

  // ── Allocate GPU buffers ─────────────────────────────────────────────────
  float* d_signal;
  cufftComplex* d_fft_out;
  float* d_psd_frame;
  float* d_psd_avg;
  DetectionResult* d_results;

  cudaMalloc(&d_signal,    kNfft * sizeof(float));
  cudaMalloc(&d_fft_out,   num_bins * sizeof(cufftComplex));
  cudaMalloc(&d_psd_frame, num_bins * sizeof(float));
  cudaMalloc(&d_psd_avg,   num_bins * sizeof(float));
  cudaMalloc(&d_results,   sizeof(DetectionResult));
  cudaMemset(d_psd_avg, 0, num_bins * sizeof(float));

  // Mode bins: 1T only for this benchmark.
  const std::vector<int> h_mode_bins = {60};
  int* d_mode_bins;
  cudaMalloc(&d_mode_bins, sizeof(int));
  cudaMemcpy(d_mode_bins, h_mode_bins.data(), sizeof(int),
             cudaMemcpyHostToDevice);

  // ── cuFFT plan ───────────────────────────────────────────────────────────
  cufftHandle plan;
  cufftPlan1d(&plan, kNfft, CUFFT_R2C, 1);

  const dim3 block(kBlockSize);
  const dim3 grid_signal((kNfft + kBlockSize - 1) / kBlockSize);
  const dim3 grid_bins((num_bins + kBlockSize - 1) / kBlockSize);

  DetectorParameters det_params;
  det_params.num_bins = num_bins;
  det_params.sample_rate_hz = kFs;
  det_params.fft_length = kNfft;
  det_params.sigma_threshold = 3.0f;
  det_params.peak_search_half_width = 3;

  // ── Warmup ───────────────────────────────────────────────────────────────
  cudaMemcpy(d_signal, h_signal.data(), kNfft * sizeof(float),
             cudaMemcpyHostToDevice);
  apply_hann_window_kernel<<<grid_signal, block>>>(d_signal, kNfft);
  cufftExecR2C(plan, d_signal, d_fft_out);
  compute_psd_kernel<<<grid_bins, block>>>(
    d_psd_frame, d_fft_out, kNfft, kHannWindowPower * kNfft);
  cudaDeviceSynchronize();

  // ── Timed loop ───────────────────────────────────────────────────────────
  const auto t_start = std::chrono::high_resolution_clock::now();

  for (int frame = 1; frame <= kNumFrames; ++frame)
  {
    cudaMemcpy(d_signal, h_signal.data(), kNfft * sizeof(float),
               cudaMemcpyHostToDevice);
    apply_hann_window_kernel<<<grid_signal, block>>>(d_signal, kNfft);
    cufftExecR2C(plan, d_signal, d_fft_out);
    compute_psd_kernel<<<grid_bins, block>>>(
      d_psd_frame, d_fft_out, kNfft, kHannWindowPower * kNfft);
    accumulate_psd_kernel<<<grid_bins, block>>>(
      d_psd_avg, d_psd_frame, num_bins, frame);
  }

  cudaDeviceSynchronize();
  const auto t_end = std::chrono::high_resolution_clock::now();

  // ── Final detection on averaged PSD ──────────────────────────────────────
  detect_instability_peaks_kernel<<<1, 1>>>(
    d_results, d_psd_avg, d_mode_bins, 1, det_params);
  cudaDeviceSynchronize();

  // ── Results ──────────────────────────────────────────────────────────────
  const double elapsed_s =
    std::chrono::duration<double>(t_end - t_start).count();
  const double us_per_frame = 1e6 * elapsed_s / kNumFrames;
  const double frames_per_s = kNumFrames / elapsed_s;
  const double samples_per_s = frames_per_s * kNfft;

  DetectionResult h_result;
  cudaMemcpy(&h_result, d_results, sizeof(DetectionResult),
             cudaMemcpyDeviceToHost);

  printf("Timing (%d frames):\n", kNumFrames);
  printf("  Total elapsed:    %.3f s\n", elapsed_s);
  printf("  Per frame:        %.2f µs\n", us_per_frame);
  printf("  Throughput:       %.0f frames/s  (%.2e samples/s)\n\n",
         frames_per_s, samples_per_s);

  printf("Detection result (after %d averaged frames):\n", kNumFrames);
  printf("  Mode freq:        %.1f Hz\n", h_result.mode_frequency_hz);
  printf("  Peak PSD:         %.4f\n", h_result.peak_psd);
  printf("  Noise floor:      %.4f\n", h_result.noise_floor);
  printf("  SNR:              %.2f sigma\n", h_result.snr_sigma);
  printf("  Instability:      %s\n",
         h_result.instability_detected ? "DETECTED" : "not detected");

  cufftDestroy(plan);
  cudaFree(d_signal);
  cudaFree(d_fft_out);
  cudaFree(d_psd_frame);
  cudaFree(d_psd_avg);
  cudaFree(d_results);
  cudaFree(d_mode_bins);

  return 0;
}
