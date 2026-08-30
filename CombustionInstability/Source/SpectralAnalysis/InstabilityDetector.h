#ifndef COMBUSTION_INSTABILITY_SPECTRAL_ANALYSIS_INSTABILITY_DETECTOR_H
#define COMBUSTION_INSTABILITY_SPECTRAL_ANALYSIS_INSTABILITY_DETECTOR_H

#include <cuda_runtime.h>

namespace CombustionInstability
{
namespace SpectralAnalysis
{

//------------------------------------------------------------------------------
/// \brief Parameters describing known acoustic mode frequencies of a combustion
///        chamber, and the detection threshold above noise floor.
///
/// Frequencies are indexed by their bin in the PSD array:
///   bin_k = round(f_mode * N_fft / f_sample)
///
/// For a LOX/CH4 chamber (Raptor-scale, subscale):
///   c ~ 1000 m/s (hot gas), R ~ 50 mm
///   f_1T = 1.841 * c / (2*pi*R) ≈ 5860 Hz
///   f_2T = 3.054 * c / (2*pi*R) ≈ 9720 Hz
//------------------------------------------------------------------------------
struct DetectorParameters
{
  //----------------------------------------------------------------------------
  /// Number of PSD bins (= N_fft/2 + 1).
  //----------------------------------------------------------------------------
  int num_bins;

  //----------------------------------------------------------------------------
  /// Sample rate of the pressure signal (Hz).
  //----------------------------------------------------------------------------
  float sample_rate_hz;

  //----------------------------------------------------------------------------
  /// FFT length used to compute the PSD.
  //----------------------------------------------------------------------------
  int fft_length;

  //----------------------------------------------------------------------------
  /// Detection threshold multiplier above noise floor (sigma units).
  /// A value of 3.0 gives ~3-sigma significance.
  //----------------------------------------------------------------------------
  float sigma_threshold;

  //----------------------------------------------------------------------------
  /// Half-width (in bins) of the local neighbourhood searched for a peak.
  //----------------------------------------------------------------------------
  int peak_search_half_width;

  //----------------------------------------------------------------------------
  /// Returns the PSD bin index closest to a given frequency.
  //----------------------------------------------------------------------------
  __host__ __device__ int frequency_to_bin(const float frequency_hz) const
  {
    return static_cast<int>(
      frequency_hz * static_cast<float>(fft_length) / sample_rate_hz + 0.5f);
  }

  //----------------------------------------------------------------------------
  /// Returns the frequency (Hz) corresponding to a given PSD bin.
  //----------------------------------------------------------------------------
  __host__ __device__ float bin_to_frequency(const int bin) const
  {
    return static_cast<float>(bin) * sample_rate_hz /
      static_cast<float>(fft_length);
  }
};

//------------------------------------------------------------------------------
/// \brief Output of the instability detection kernel: one entry per monitored
///        mode frequency.
//------------------------------------------------------------------------------
struct DetectionResult
{
  //----------------------------------------------------------------------------
  /// Frequency of the monitored mode (Hz).
  //----------------------------------------------------------------------------
  float mode_frequency_hz;

  //----------------------------------------------------------------------------
  /// PSD value at the detected peak bin.
  //----------------------------------------------------------------------------
  float peak_psd;

  //----------------------------------------------------------------------------
  /// Noise floor estimate (mean PSD in a band around the mode, excluding peak).
  //----------------------------------------------------------------------------
  float noise_floor;

  //----------------------------------------------------------------------------
  /// Signal-to-noise ratio in sigma units: (peak - noise_floor) / noise_std.
  //----------------------------------------------------------------------------
  float snr_sigma;

  //----------------------------------------------------------------------------
  /// True if snr_sigma > sigma_threshold.
  //----------------------------------------------------------------------------
  bool instability_detected;
};

//------------------------------------------------------------------------------
/// \brief Detect instability peaks in a PSD array near a set of target modes.
///
/// For each monitored mode frequency, searches a ±peak_search_half_width bin
/// neighbourhood for the maximum PSD value, estimates the local noise floor
/// from the surrounding region, and computes the SNR in sigma units.
///
/// \param[out] results       Device pointer to DetectionResult array, one per mode.
/// \param[in]  psd           Device pointer to PSD array (length num_bins).
/// \param[in]  mode_bins     Device pointer to target mode bin indices (length num_modes).
/// \param[in]  num_modes     Number of modes to monitor.
/// \param[in]  params        Detector parameters.
//------------------------------------------------------------------------------
__global__ void detect_instability_peaks_kernel(
  DetectionResult* results,
  const float* psd,
  const int* mode_bins,
  const int num_modes,
  const DetectorParameters params)
{
  const int mode_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (mode_idx >= num_modes) { return; }

  const int target_bin = mode_bins[mode_idx];
  const int hw = params.peak_search_half_width;

  // ── Find peak in [target_bin - hw, target_bin + hw] ──────────────────────
  float peak_psd = 0.0f;
  int peak_bin = target_bin;
  for (int b = target_bin - hw; b <= target_bin + hw; ++b)
  {
    if (b >= 0 && b < params.num_bins && psd[b] > peak_psd)
    {
      peak_psd = psd[b];
      peak_bin = b;
    }
  }

  // ── Estimate noise floor from a wider band, excluding the peak region ──────
  // Use bins in [target - 4*hw, target + 4*hw] excluding [target - hw, target + hw].
  const int noise_hw = 4 * hw;
  float noise_sum = 0.0f;
  float noise_sq_sum = 0.0f;
  int noise_count = 0;

  for (int b = target_bin - noise_hw; b <= target_bin + noise_hw; ++b)
  {
    if (b < 0 || b >= params.num_bins) { continue; }
    if (b >= target_bin - hw && b <= target_bin + hw) { continue; }

    noise_sum += psd[b];
    noise_sq_sum += psd[b] * psd[b];
    ++noise_count;
  }

  const float noise_floor = (noise_count > 0)
    ? noise_sum / static_cast<float>(noise_count)
    : 0.0f;

  const float noise_var = (noise_count > 1)
    ? (noise_sq_sum / noise_count - noise_floor * noise_floor)
    : 0.0f;
  const float noise_std = sqrtf(fmaxf(noise_var, 0.0f));

  const float snr = (noise_std > 0.0f)
    ? (peak_psd - noise_floor) / noise_std
    : 0.0f;

  DetectionResult result;
  result.mode_frequency_hz = params.bin_to_frequency(target_bin);
  result.peak_psd = peak_psd;
  result.noise_floor = noise_floor;
  result.snr_sigma = snr;
  result.instability_detected = (snr > params.sigma_threshold);

  results[mode_idx] = result;
}

} // namespace SpectralAnalysis
} // namespace CombustionInstability

#endif // COMBUSTION_INSTABILITY_SPECTRAL_ANALYSIS_INSTABILITY_DETECTOR_H
