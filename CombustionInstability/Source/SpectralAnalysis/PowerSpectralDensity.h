#ifndef COMBUSTION_INSTABILITY_SPECTRAL_ANALYSIS_POWER_SPECTRAL_DENSITY_H
#define COMBUSTION_INSTABILITY_SPECTRAL_ANALYSIS_POWER_SPECTRAL_DENSITY_H

#include <cufft.h>
#include <cuda_runtime.h>

namespace CombustionInstability
{
namespace SpectralAnalysis
{

//------------------------------------------------------------------------------
/// \brief Apply a Hann window to a real time-series segment in-place.
///
/// w(n) = 0.5 * (1 - cos(2π n / N))
///
/// \param[in,out] signal  Device pointer to real samples, length N.
/// \param[in]     N       Window length (number of samples).
//------------------------------------------------------------------------------
__global__ void apply_hann_window_kernel(float* signal, const int N)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) { return; }

  const float w = 0.5f * (1.0f - cosf(2.0f * M_PI * n / N));
  signal[n] *= w;
}

//------------------------------------------------------------------------------
/// \brief Compute power spectral density magnitudes from cuFFT R2C output.
///
/// PSD(k) = |X(k)|² / (N * W)
///
/// where W = sum(w²) is the window power normalisation factor.
/// For a Hann window of length N: W = N * 3/8 ≈ 0.375 * N.
///
/// The R2C output has N/2+1 complex bins. This kernel produces N/2+1 real PSD
/// values, suitable for one-sided power spectrum.
///
/// \param[out] psd       Device pointer to PSD output, length N/2+1.
/// \param[in]  fft_out   Device pointer to cuFFT cufftComplex output, length N/2+1.
/// \param[in]  N         FFT length (number of input samples).
/// \param[in]  window_power  Normalisation: sum of squared window weights.
//------------------------------------------------------------------------------
__global__ void compute_psd_kernel(
  float* psd,
  const cufftComplex* fft_out,
  const int N,
  const float window_power)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_bins = N / 2 + 1;
  if (k >= num_bins) { return; }

  const float re = fft_out[k].x;
  const float im = fft_out[k].y;
  // Two-sided correction: multiply non-DC, non-Nyquist bins by 2 for one-sided.
  const float two_sided_factor = (k > 0 && k < num_bins - 1) ? 2.0f : 1.0f;
  psd[k] = two_sided_factor * (re * re + im * im) / (N * window_power);
}

//------------------------------------------------------------------------------
/// \brief Accumulate (average) a new PSD frame into a running mean.
///
/// running_mean = running_mean + (new_psd - running_mean) / frame_count
///
/// Using Welch's online average to avoid overflow for large frame counts.
///
/// \param[in,out] running_mean  Device pointer, length N/2+1.
/// \param[in]     new_psd       Device pointer, length N/2+1.
/// \param[in]     num_bins      N/2+1.
/// \param[in]     frame_count   Current frame index (1-based).
//------------------------------------------------------------------------------
__global__ void accumulate_psd_kernel(
  float* running_mean,
  const float* new_psd,
  const int num_bins,
  const int frame_count)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_bins) { return; }

  running_mean[k] += (new_psd[k] - running_mean[k]) / frame_count;
}

//------------------------------------------------------------------------------
/// \brief Hann window power normalisation factor W = sum(w²) / N.
///
/// For a Hann window: W = 3/8 = 0.375 (independent of N).
//------------------------------------------------------------------------------
constexpr float kHannWindowPower = 0.375f;

} // namespace SpectralAnalysis
} // namespace CombustionInstability

#endif // COMBUSTION_INSTABILITY_SPECTRAL_ANALYSIS_POWER_SPECTRAL_DENSITY_H
