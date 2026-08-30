"""
Combustion Instability Detection in JAX.

Implements the same FFT-based PSD pipeline as the CUDA C++ version, using JAX
for GPU-accelerated computation with JIT compilation and autograd.

Physics background: documents/CombustionInstability.md

Demonstrates:
  1. Signal synthesis (noise + instability tone at 1T mode frequency)
  2. Hann-windowed PSD estimation via jnp.fft.rfft (Welch's method)
  3. Peak detection and SNR estimation
  4. Comparison: CUDA C++ approach vs JAX approach
  5. JAX advantage: autograd lets us optimize a notch filter to damp instability

Usage (inside the PropulsionWithCUDA Docker container):
    python3 Python/combustion_instability_jax.py

Dependencies: jax[cuda13-local] (installed in the Docker image)
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple


# ── Physical constants for LOX/CH4 subscale chamber ──────────────────────────

SAMPLE_RATE_HZ = 100_000.0   # 100 kHz pressure sensor
N_FFT = 1024                  # FFT length → Δf = 97.7 Hz
CHAMBER_RADIUS_M = 0.05       # 50 mm radius (subscale)
SPEED_OF_SOUND_MS = 1000.0    # ~1000 m/s in hot LOX/CH4 combustion gas

# Bessel function zeros for transverse mode frequencies:
#   f_nT = alpha_n * c / (2π R)
ALPHA_1T = 1.84118            # first zero of J_1'
ALPHA_2T = 3.05424

F_1T_HZ = ALPHA_1T * SPEED_OF_SOUND_MS / (2.0 * np.pi * CHAMBER_RADIUS_M)
F_2T_HZ = ALPHA_2T * SPEED_OF_SOUND_MS / (2.0 * np.pi * CHAMBER_RADIUS_M)

print(f"1T mode: {F_1T_HZ:.1f} Hz  |  2T mode: {F_2T_HZ:.1f} Hz")


# ── Signal synthesis ──────────────────────────────────────────────────────────

def synthesize_pressure_signal(
    key: jax.Array,
    n_samples: int,
    sample_rate: float,
    instability_freq: float,
    instability_amplitude: float,
    noise_amplitude: float,
) -> jax.Array:
    """
    Generate a synthetic chamber pressure time series.

    p(t) = noise_amplitude * N(0,1) + instability_amplitude * sin(2π f_inst t)

    Returns: [n_samples] float32
    """
    t = jnp.arange(n_samples) / sample_rate
    noise = noise_amplitude * jax.random.normal(key, shape=(n_samples,))
    tone = instability_amplitude * jnp.sin(2.0 * jnp.pi * instability_freq * t)
    return noise + tone


# ── Hann window ───────────────────────────────────────────────────────────────

def hann_window(n: int) -> jax.Array:
    """Hann window of length n. Power W = 3/8 = 0.375."""
    k = jnp.arange(n)
    return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * k / n))


# ── PSD estimation ────────────────────────────────────────────────────────────

@jax.jit
def compute_psd_frame(signal: jax.Array) -> jax.Array:
    """
    Compute one-sided PSD for a single windowed frame.

    PSD(k) = |X(k)|² / (N * W)

    where W = sum(w²) / N ≈ 0.375 for Hann window.

    Args:
        signal: [N] real-valued time series (one FFT frame)

    Returns:
        psd: [N//2 + 1] one-sided PSD
    """
    n = signal.shape[0]
    w = hann_window(n)
    windowed = signal * w

    X = jnp.fft.rfft(windowed)               # [N//2 + 1] complex
    power = jnp.abs(X) ** 2

    window_power = jnp.sum(w ** 2)           # ≈ 0.375 * N for Hann

    # Two-sided → one-sided correction (double non-DC, non-Nyquist bins).
    num_bins = n // 2 + 1
    two_sided_factor = jnp.ones(num_bins)
    two_sided_factor = two_sided_factor.at[1:num_bins - 1].set(2.0)

    return two_sided_factor * power / (n * window_power)


def welch_psd(
    signal: jax.Array,
    n_fft: int,
    overlap: float = 0.5,
) -> tuple[jax.Array, jax.Array]:
    """
    Welch's averaged PSD estimate.

    Args:
        signal: [N_total] time series
        n_fft: FFT length
        overlap: fractional overlap between windows (0.5 = 50%)

    Returns:
        psd_mean: [n_fft//2 + 1] averaged PSD
        freqs:    [n_fft//2 + 1] frequency axis (Hz)
    """
    step = int(n_fft * (1.0 - overlap))
    n_total = signal.shape[0]

    starts = jnp.arange(0, n_total - n_fft + 1, step)
    n_frames = starts.shape[0]

    def get_frame(start):
        return signal[start:start + n_fft]

    # vmap over frames for parallel PSD computation.
    frames = jax.vmap(get_frame)(starts)    # [n_frames, n_fft]
    psds = jax.vmap(compute_psd_frame)(frames)  # [n_frames, n_fft//2+1]

    psd_mean = jnp.mean(psds, axis=0)

    freqs = jnp.fft.rfftfreq(n_fft) * SAMPLE_RATE_HZ

    return psd_mean, freqs


# ── Instability detection ─────────────────────────────────────────────────────

class DetectionResult(NamedTuple):
    mode_frequency_hz: float
    peak_psd: float
    noise_floor: float
    snr_sigma: float
    instability_detected: bool


def detect_instability(
    psd: jax.Array,
    freqs: jax.Array,
    mode_frequency_hz: float,
    sigma_threshold: float = 3.0,
    search_band_hz: float = 500.0,
    noise_band_hz: float = 2000.0,
) -> DetectionResult:
    """
    Detect an instability mode peak in a PSD.

    Searches ±search_band_hz around mode_frequency_hz for a peak,
    estimates noise from a ±noise_band_hz band (excluding the search band),
    and computes SNR in sigma units.
    """
    df = float(freqs[1] - freqs[0])

    search_hw_bins = int(search_band_hz / df)
    noise_hw_bins = int(noise_band_hz / df)

    target_bin = int(mode_frequency_hz / df)

    # Search band for peak.
    lo = max(0, target_bin - search_hw_bins)
    hi = min(psd.shape[0] - 1, target_bin + search_hw_bins)
    search_psd = psd[lo:hi + 1]
    peak_psd = float(jnp.max(search_psd))

    # Noise band: exclude search band.
    noise_lo = max(0, target_bin - noise_hw_bins)
    noise_hi = min(psd.shape[0] - 1, target_bin + noise_hw_bins)
    noise_psd = jnp.concatenate([psd[noise_lo:lo], psd[hi + 1:noise_hi + 1]])

    noise_floor = float(jnp.mean(noise_psd))
    noise_std = float(jnp.std(noise_psd))

    snr = (peak_psd - noise_floor) / noise_std if noise_std > 0 else 0.0

    return DetectionResult(
        mode_frequency_hz=mode_frequency_hz,
        peak_psd=peak_psd,
        noise_floor=noise_floor,
        snr_sigma=snr,
        instability_detected=snr > sigma_threshold,
    )


# ── JAX advantage: notch filter optimization via autograd ────────────────────

def apply_notch_filter(signal: jax.Array, freq_hz: float, q_factor: float) -> jax.Array:
    """
    Apply a simple FIR notch filter at freq_hz with quality factor q_factor.

    This is a 3-tap FIR approximation:  y[n] = x[n] - alpha * (x[n-1] + x[n+1])
    where alpha is learned. Demonstrates JAX autodiff through signal processing.
    """
    omega = 2.0 * jnp.pi * freq_hz / SAMPLE_RATE_HZ
    alpha = 0.5 * jnp.cos(omega)
    filtered = signal.at[1:-1].add(-alpha * (signal[:-2] + signal[2:]))
    return filtered


def notch_loss(alpha: jax.Array, signal: jax.Array, target_freq_hz: float) -> jax.Array:
    """
    Loss = PSD at target frequency after applying notch with coefficient alpha.
    Minimise to drive the instability mode power to zero.
    """
    omega = 2.0 * jnp.pi * target_freq_hz / SAMPLE_RATE_HZ
    filtered = signal.at[1:-1].add(-alpha * (signal[:-2] + signal[2:]))
    psd = compute_psd_frame(filtered[:N_FFT])
    target_bin = int(target_freq_hz * N_FFT / SAMPLE_RATE_HZ)
    return psd[target_bin]


notch_loss_grad = jax.jit(jax.value_and_grad(notch_loss))


def optimize_notch_filter(
    signal: jax.Array,
    target_freq_hz: float,
    learning_rate: float = 0.1,
    n_steps: int = 50,
) -> tuple[float, list[float]]:
    """
    Gradient-descent optimisation of notch filter coefficient alpha.

    Returns: (optimal_alpha, loss_history)

    This is the unique JAX capability: autodiff through the FFT pipeline.
    CUDA C++ would require hand-derived gradients.
    """
    alpha = jnp.array(0.0)
    losses = []

    for _ in range(n_steps):
        loss, grad = notch_loss_grad(alpha, signal, target_freq_hz)
        alpha = alpha - learning_rate * grad
        losses.append(float(loss))

    return float(alpha), losses


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_psd(signal: jax.Array, repeats: int = 100) -> float:
    """Time Welch PSD computation. Returns avg ms per call."""
    # Warmup
    psd, freqs = welch_psd(signal, N_FFT)
    jax.block_until_ready(psd)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        psd, freqs = welch_psd(signal, N_FFT)
        jax.block_until_ready(psd)
        times.append(time.perf_counter() - t0)

    return 1e3 * sum(times) / len(times)


def main() -> None:
    print(f"\nJAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}\n")

    key = jax.random.PRNGKey(0)

    # ── Synthesize 0.1-second pressure signal ─────────────────────────────
    n_samples = int(SAMPLE_RATE_HZ * 0.1)  # 10,000 samples at 100 kHz
    signal = synthesize_pressure_signal(
        key, n_samples,
        sample_rate=SAMPLE_RATE_HZ,
        instability_freq=F_1T_HZ,
        instability_amplitude=5.0,
        noise_amplitude=1.0,
    )
    jax.block_until_ready(signal)
    print(f"Signal: {n_samples} samples at {SAMPLE_RATE_HZ:.0f} Hz")
    print(f"Injected 1T instability at {F_1T_HZ:.1f} Hz\n")

    # ── PSD via Welch's method ────────────────────────────────────────────
    psd, freqs = welch_psd(signal, N_FFT, overlap=0.5)
    jax.block_until_ready(psd)
    print(f"PSD bins: {psd.shape[0]}  Δf = {float(freqs[1]):.2f} Hz")

    # ── Detect 1T and 2T modes ────────────────────────────────────────────
    for mode_name, mode_freq in [("1T", F_1T_HZ), ("2T", F_2T_HZ)]:
        result = detect_instability(psd, freqs, mode_freq)
        status = "DETECTED" if result.instability_detected else "not detected"
        print(f"  {mode_name} ({mode_freq:.0f} Hz): SNR={result.snr_sigma:.1f}σ  "
              f"peak={result.peak_psd:.4f}  noise={result.noise_floor:.4f}  "
              f"→ {status}")

    # ── JAX-only: optimize notch filter via autograd ──────────────────────
    print(f"\nOptimizing notch filter at {F_1T_HZ:.1f} Hz via JAX autograd...")
    alpha_opt, losses = optimize_notch_filter(signal, F_1T_HZ, n_steps=50)
    print(f"  Optimal alpha: {alpha_opt:.4f}")
    print(f"  Initial loss: {losses[0]:.4f}  →  Final loss: {losses[-1]:.4f}  "
          f"(reduction: {100*(1 - losses[-1]/losses[0]):.1f}%)")

    # ── Benchmark ─────────────────────────────────────────────────────────
    avg_ms = benchmark_psd(signal)
    frames_per_s = 1e3 / avg_ms
    print(f"\nPSD pipeline (Welch, {N_FFT}-pt FFT, 50% overlap, "
          f"{(n_samples - N_FFT) // (N_FFT // 2) + 1} frames):")
    print(f"  {avg_ms:.3f} ms/call  →  {frames_per_s:.0f} calls/s")
    print(f"  Effective sample throughput: {frames_per_s * n_samples:.2e} samples/s")


if __name__ == '__main__':
    main()
