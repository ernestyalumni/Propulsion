# Combustion Instability Detection: Physics and Math Reference

## What Is Combustion Instability?

Combustion instability is a **positive feedback loop** between unsteady heat release
in a combustion chamber and the chamber's acoustic resonance modes. When the two
couple, pressure oscillations grow exponentially until they either saturate
nonlinearly or destroy the engine.

This is not a niche failure mode. It caused the Saturn V F-1 engine to require
years of empirical damping work. SpaceX Raptor (LOX/CH4) deals with
high-frequency screech. Blue Origin BE-4 encountered it during development.

## The Rayleigh Criterion (1878)

Lord Rayleigh's criterion is the fundamental instability condition:

```
∮ p'(x,t) · q'(x,t) dV dt > 0
```

where:
- p'(x,t) = acoustic pressure fluctuation
- q'(x,t) = heat release rate fluctuation
- The integral is over the combustion volume and one oscillation period

**Physical interpretation:** If heat is added *in phase* with pressure (positive
correlation), energy is pumped into the acoustic field → instability.
If heat is added *out of phase* with pressure → acoustic damping.

This is thermoacoustic feedback. The same physics governs Rijke tubes,
gas turbine combustors, and rocket engines.

## Acoustic Modes of a Cylindrical Combustion Chamber

A cylindrical chamber (radius R, length L) supports multiple resonant modes.
The mode frequencies are determined by the wave equation with appropriate BCs.

**Longitudinal modes (1L, 2L, ...):**
```
f_nL = n * c / (2L)      n = 1, 2, 3, ...
```

**Transverse modes — tangential (1T, 2T, ...):**
```
f_nT = alpha_n * c / (2*pi*R)
```
where alpha_n are zeros of the derivative of Bessel functions J_m':
- 1T mode: alpha ≈ 1.841  (first tangential)
- 2T mode: alpha ≈ 3.054
- 1R mode: alpha ≈ 3.832  (first radial)

**Combined modes (1T1L, etc.):** coupling of transverse and longitudinal.

**Nomenclature:** High-frequency instability = transverse modes (dangerous,
can destroy an injector face in milliseconds). Low-frequency = longitudinal
or bulk modes (chugging, ~10-400 Hz). Medium-frequency = coupled chamber-feed
system (buzzing, 400-1000 Hz).

### Example: LOX/CH4 Chamber (Raptor-scale)

For a subscale LOX/CH4 combustor with R ≈ 50 mm and speed of sound c ≈ 1000 m/s
(hot combustion gas, ~3500 K):
```
f_1T = 1.841 * 1000 / (2*pi*0.05) ≈ 5860 Hz
f_2T = 3.054 * 1000 / (2*pi*0.05) ≈ 9700 Hz
```

## Detection via Power Spectral Density

A dynamic pressure sensor in the chamber wall produces a time series p(t)
sampled at frequency f_s. Instability shows up as a **sharp spectral peak**
at or near a known acoustic mode frequency.

**Power Spectral Density (PSD) estimation via FFT (Welch's method):**

1. Divide p(t) into M overlapping windows of length N_fft.
2. Apply Hann window: w(n) = 0.5 * (1 - cos(2π n / N_fft))
3. FFT each window: P_k = FFT{w * p}
4. Average the squared magnitudes: PSD(f_k) = (1/M) * |P_k|² / (f_s * sum(w²))
5. Frequency resolution: Δf = f_s / N_fft

**Frequency axis:** f_k = k * f_s / N_fft for k = 0,..., N_fft/2

**Detection threshold:** a peak at frequency f is an instability candidate if:
```
PSD(f) > mu_noise + sigma_threshold * sigma_noise
```
where mu_noise and sigma_noise are estimated from the baseline (pre-test) PSD.

## Instability Precursors

The system undergoes a bifurcation from stable → limit cycle oscillation.
Before the bifurcation, detectable precursors appear:

1. **Decreasing permutation entropy** — the pressure signal becomes more
   deterministic (structured) as the instability approaches.
2. **Growing autocorrelation length** — critical slowing-down near the bifurcation.
3. **PSD peak growth** — the mode peak rises above the noise floor tens to
   hundreds of milliseconds before onset.

The early warning window is typically **100–500 ms** before full onset.
This is the operating envelope for a real-time detection system.

## GPU Acceleration Motivation

At f_s = 100 kHz with N_fft = 1024:
- Frequency resolution: 97.7 Hz — resolves combustion modes
- One FFT on 1024 samples: ~microseconds on GPU vs ~milliseconds on CPU
- For M = 100 overlapping windows, full PSD: GPU enables real-time at f_s

For a Raptor-scale engine with 100+ pressure sensors and multi-kHz sampling,
the total data rate can exceed 100 MB/s. GPU parallelism is necessary for
simultaneous multi-channel real-time detection.

## The Connection to Radar Matched Filtering

The cross-correlation between the measured signal and a reference waveform is:
```
(x ⋆ y)[τ] = Σ_t x[t] * y[t+τ]
```

By the convolution theorem:
```
(x ⋆ y) = IFFT{FFT{x}* · FFT{y}}
```

Radar matched filtering correlates the received signal against the transmitted
pulse template to detect its time-of-arrival. Combustion instability detection
correlates the pressure signal's spectrum against the theoretical mode template
(a delta at the predicted mode frequency) to detect the instability.

**Same mathematical operation. Different physical domain.**

## Signal Synthesis for Testing

A synthetic test signal with a 1T instability:
```python
t = np.arange(N) / f_s
p = noise_amplitude * rng.normal(size=N)          # broadband combustion noise
p += instability_amplitude * np.sin(2*pi*f_1T*t)  # 1T mode injection
```

The SNR of the instability in the PSD grows as more FFT windows are averaged.
Early detection requires distinguishing a growing peak from noise before SNR >> 1.

## References

### Primary Textbooks (read in this order)

---

**[1] Rayleigh, Lord (J. W. Strutt) (1878).
*The Theory of Sound.* Volumes I and II.
Macmillan and Co., London.
(Reprinted by Dover Publications, 1945.)**

The origin of thermoacoustic instability theory. In Chapter XVII Rayleigh gives
his famous criterion: "If heat be periodically communicated to, and abstracted
from, a mass of air vibrating (for example) in a cylinder bounded by a piston,
the effect produced will depend upon the phase of the vibration at which the
transfer of heat takes place. If heat be given to the air at the moment of
greatest condensation, or be taken from it at the moment of greatest rarefaction,
the vibration is encouraged."

In modern notation: ∮ p'q' dV dt > 0 is the instability condition.

Also relevant: Rayleigh (1878). "The explanation of certain acoustical
phenomena." *Nature* 18, 319–321. (Shorter paper, same criterion.)

**What to read:** For the criterion in its original form, *Nature* 18 is
2 pages. For the full acoustic theory background, Chapters I, IX, XVII of
*The Theory of Sound* Vol. II.

---

**[2] Yang, V. and Anderson, W. E. (Eds.) (1995).
*Liquid Rocket Engine Combustion Instability.*
Progress in Astronautics and Aeronautics, Vol. 169.
AIAA, Washington, DC. 577 pp.**

AIAA catalog: https://arc.aiaa.org/doi/book/10.2514/4.866371

The first comprehensive U.S. treatment of liquid rocket combustion instability
since NASA SP-194 (1972). Edited volume with contributions from leading
engineers who worked on Saturn V F-1, Space Shuttle SSME, and other programs.

Organized into four parts:
1. **Phenomenology and Case Studies** — historical engine failures, visual
   observations from windowed combustors, instrumentation.
2. **Fundamental Mechanisms** — injector-coupled instability, droplet
   combustion dynamics, acoustic coupling of spray flames.
3. **Stability Analysis** — linear and nonlinear stability methods, Galerkin
   expansion on acoustic modes, N-tau (sensitive time lag) model.
4. **Testing** — bomb testing, cold-flow testing, HIFREQ instrumentation.

**What to read for mode analysis:** Chapter 2 (Zinn, "Acoustic Waves in Gas-Filled
Cylinders") for the Bessel function mode analysis. Chapter 6 (Culick, "Stability
of Acoustic Modes") for the linearized equations.

**Historical significance:** This volume documents the Saturn V F-1 engine
instability (1962–1965), during which 2000 full-scale engine tests were run
and acoustic baffles were empirically discovered as a solution. It remains the
primary engineering reference for hardware-level context.

---

**[3] Culick, F. E. C. (2006).
*Unsteady Motions in Combustion Chambers for Propulsion Systems.*
NATO Research and Technology Organisation, AGARDograph RTO-AG-AVT-039.
Available open-access via NATO STO.**

URL (open access PDF): https://publications.sto.nato.int/publications/STO%20Technical%20Reports/RTO-AG-AVT-039/$$AG-AVT-039-ALL.pdf

NATO STO page: https://www.sto.nato.int/document/unsteady-motions-in-combustion-chambers-for-propulsion-systems/

Fred Culick (Caltech) spent 40+ years developing the theoretical foundation
for rocket combustion instability. This monograph is the comprehensive
synthesis, covering:

- **Volume-averaged equations:** the two-energy, two-pressure formulation
  that separates acoustic modes from mean flow.
- **Galerkin expansion:** expressing chamber pressure as a sum of acoustic
  eigenfunctions; reduces the PDE to a system of coupled ODEs for modal
  amplitudes.
- **Linear stability:** the growth rate of each mode is the balance of
  driving (Rayleigh integral) minus damping (viscous, injection, nozzle).
- **Nonlinear dynamics:** limit-cycle amplitudes, triggering (subcritical
  bifurcation), combustion noise modeling.
- **N-tau model:** the sensitive time-lag model (Crocco & Cheng, 1956) and
  its generalization; relates injector response function to instability.

**What to read:** Chapter 1 (overview), Chapter 3 (acoustic equations for
cylinders — the Bessel function mode derivation is here), Chapter 7 (Rayleigh
criterion in the Galerkin framework). The full monograph is ~700 pages; it is
a reference, not cover-to-cover reading.

---

**[4] Lieuwen, T. C. (2012).
*Unsteady Combustor Physics.*
Cambridge University Press, New York. 424 pp.
ISBN: 978-1-107-01599-9**

Publisher page: https://www.cambridge.org/core/books/unsteady-combustor-physics/

The modern graduate textbook. More accessible than Culick [3] because it is
organized as a coherent pedagogical sequence rather than a monograph. Covers
gas turbine and rocket combustors both.

Chapter structure relevant to our work:
- **Ch. 2** — Acoustic wave equation, boundary conditions, eigenmodes in
  cylinders (Bessel zeros). This is where f_nT = α_n c / (2πR) is derived.
- **Ch. 5** — The Rayleigh criterion: formal derivation from energy balance,
  not just the intuitive statement.
- **Ch. 7** — Flame transfer functions: how heat release responds to acoustic
  forcing. This is the "q" in the Rayleigh integral.
- **Ch. 10** — Nonlinear dynamics: limit cycles, triggering, hysteresis.
- **Ch. 12** — Sensing and instrumentation: pressure sensors, optical methods,
  what a real detection system looks like.

**What to read first:** Ch. 2 (modes) + Ch. 5 (Rayleigh). These two chapters
provide the mathematical foundation for everything in our GPU code.

---

### Review Papers

---

**[5] Lieuwen, T. (2003).
*Modeling premixed combustion-acoustic wave interactions: A review.*
Journal of Propulsion and Power, 19(5), 765–781.**

AIAA: https://arc.aiaa.org/doi/10.2514/2.6197

A focused review of how premixed flames couple to acoustic oscillations. Covers:
- Acoustic boundary conditions for confined flames
- Flame transfer function (FTF) measurement and modeling
- Linear stability criterion in terms of the FTF
- Connection between chemical kinetics and the time delay τ in the N-tau model

Accessible (~16 pages, JPP format) and is a good companion to Ch. 5 of [4].

---

**[6] Lieuwen, T. and Yang, V. (Eds.) (2005).
*Combustion Instabilities in Gas Turbine Engines: Operational Experience,
Fundamental Mechanisms, and Modeling.*
Progress in Astronautics and Aeronautics, Vol. 210. AIAA.**

Companion to [2] for gas turbine rather than rocket applications. Relevant
because CHAOS Industries' radar hardware context resembles gas turbine sensors
more than rocket engines in terms of sampling rate and sensor placement.

---

**[7] Culick, F. E. C. and Yang, V. (1995).
*Overview of combustion instabilities in liquid-propellant rocket engines.*
In Yang & Anderson [2], Chapter 1, pp. 3–37.**

The introductory chapter of [2], written by Culick himself. Best 35-page
summary of the entire field: history, phenomenology, stability criteria,
control approaches. Read this before [3].

---

### Detection and Early Warning Papers

---

**[8] Kobayashi, W., Murayama, S., Hachijo, T., and Gotoda, H. (2019).
*Early Detection of Thermoacoustic Combustion Instability Using a Methodology
Combining Complex Networks and Machine Learning.*
Physical Review Applied 11, 064034.**

ResearchGate: https://www.researchgate.net/publication/333796874

Uses pressure time-series from a lean-premixed gas turbine combustor. Builds
a complex network from the permutation patterns of the pressure signal and
extracts network topology metrics (degree distribution, clustering coefficient)
that change signature ~200 ms before instability onset. These features feed
into a machine learning classifier. This paper is the direct motivation for
the precursor detection section of our code.

---

**[9] Nair, V. and Sujith, R. I. (2014).
*Multifractality in combustion noise: predicting an impending combustion
instability.*
Journal of Fluid Mechanics 747, 635–655.**

ResearchGate: https://www.researchgate.net/publication/263129024

Demonstrates that the pressure signal transitions from multifractal (turbulent
noise) to monofractal (periodic oscillation) as the system approaches a limit
cycle. The Hölder exponent distribution narrows — this is a quantitative
precursor. Related to the "critical slowing-down" mentioned in our code.

---

**[10] Gotoda, H., Hayashi, T., Hakami, A., Okuno, Y., and Tachibana, S. (2011).
*Dynamic properties of combustion instability in a lean premixed gas-turbine
combustor.*
Chaos 21, 013124.**

Shows that the permutation entropy of the pressure signal decreases
monotonically as the equivalence ratio is swept toward the unstable regime.
Permutation entropy H(m) for embedding dimension m:

```
H(m) = - sum_π p(π) * log p(π)
```

where π are the m! possible ordinal patterns of m consecutive samples. As the
signal becomes a periodic oscillation, only 2 of the m! patterns appear →
H → 0.

**Connection to our code:** the `detect_instability` function in
`InstabilityDetector.h` implements the simpler PSD-peak approach. Permutation
entropy would be a complementary precursor that triggers earlier.

---

**[11] Murayama, S. and Gotoda, H. (2021).
*Detection of precursors of combustion instability using convolutional
recurrent neural networks.*
Combustion and Flame 232, 111558.**

ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0010218021003011

A convolutional LSTM trained on the pressure time series detects instability
150–300 ms before onset with >95% precision on held-out test sets. Illustrates
the current state-of-the-art for data-driven detection systems.

---

**[12] Frontiers in Energy Research (2022).
*Analysis of Tangential Combustion Instability Modes in a LOX/Kerosene Liquid
Rocket Engine Based on OpenFOAM.*
Frontiers in Energy Research, January 2022.**

URL: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.810439/full

LES simulation of a multi-element LOX/kerosene engine showing self-excited 1T
mode at the frequency predicted by the Bessel formula. DMD (Dynamic Mode
Decomposition) of the pressure field extracts the 1T mode shape — this is the
same frequency we inject at bin 60 in our test signal.

---

### Industrial / Hardware References

---

**[13] PCB Piezotronics (Amphenol). Combustion Dynamics Pressure Sensors.**

URL: https://www.pcb.com/applications/energy/combustion-dynamics

The physical sensor technology behind real engine monitoring. PCB's Model 112
and 113 series are used in SpaceX, Blue Origin, and Aerojet Rocketdyne engines.
Key specs: frequency response 10 Hz – 100 kHz, rated to 100,000 psi peak, water-
cooled versions for chamber wall mounting.

Our simulation uses f_s = 100 kHz, consistent with these sensors.

---

**[14] GE Energy. US Patent 6,993,960 (2006) and US Patent 7,204,133 (2007).
*Combustion dynamics monitoring.*
Assignee: General Electric Company.**

The industrial signal processing chain: dynamic pressure → FFT → PSD averaging
→ threshold comparison against mode frequencies — exactly what `PowerSpectralDensity.h`
and `InstabilityDetector.h` implement. These patents describe the hardware
embodiment in GE gas turbines and are useful for understanding how the algorithm
maps to real engine monitoring systems.

---

### Reading Roadmap

For working through the physics and math together:

1. Read Culick & Yang [7] (35 pages) for context: history, failure modes, what the
   physics actually looks like in a real engine.
2. Read Lieuwen [4] Ch. 2: derive the acoustic mode frequencies from the wave equation.
   The Bessel function zeros come from the boundary condition ∂p/∂r = 0 at r = R.
3. Read Lieuwen [4] Ch. 5: derive the Rayleigh criterion from the energy equation.
   This makes ∮ p'q' dV dt > 0 rigorous, not just intuitive.
4. Read Milakov & Gimelshein [2] (our online softmax paper — different field but
   analogous math problem) to see how a two-pass algorithm becomes one-pass via
   running accumulators. The Welch PSD averaging in our GPU code uses the same pattern.
5. Read Kobayashi et al. [8] to understand what a real ML-based detection system
   looks like and how it connects to our `InstabilityDetector.h` kernel.
