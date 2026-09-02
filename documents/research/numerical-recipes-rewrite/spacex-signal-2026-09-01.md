# SpaceX capability signal — harvested 2026-09-01

Recorded in the form story 06 prescribes: extracted capability statements, the
source URL, and the retrieval date. No posting text is copied in full. Each
capability is mapped to a directory that covers it or recorded as a gap.

| Capability implied by the posting | Source (retrieved 2026-09-01) | Covered by | Gap |
|---|---|---|---|
| 6-DOF dynamics simulation models for GNC analysis and validation | https://job-boards.greenhouse.io/spacex/jobs/8559003002 (Sr. GNC Engineer, Starship) | `Cosmos/Source/Numerical/ODE/`, `Cosmos/Source/Astrodynamics/` (3-DOF only) | rigid-body attitude dynamics on SO(3) with the story-08 contract |
| Trajectory analysis, performance dispersions, flight stability | same | `Cosmos/Source/Astrodynamics/Propagators/NumerovOrbit.h` | dispersion (Monte Carlo) driver, linearized stability (eigenvalues) |
| Monte Carlo analysis of GNC software; hardware-in-the-loop testing | same | `Monoclaw/Embedded/AvionicsHIL` (HIL) | random-number and quasi-random infrastructure; batched propagation (CUDA candidate) |
| Post-flight data review and correlation with analyses | same | nothing | least-squares fitting, distribution comparison, spectral analysis of telemetry |
| Orbital mechanics, classical dynamics, aerodynamics, sensors and actuators, control systems | same | `Cosmos/` (orbits), `Cosmos/QuaternionConventionLab` (attitude representation) | aerodynamic table lookup (2-D/3-D interpolation), sensor/actuator models, control allocation |
| High-fidelity real-time simulation software used for all vehicles; models of fluids, electronics, multi-body physics; reliability and performance | https://job-boards.greenhouse.io/spacex/jobs/8603611002 (Sr. Software Engineer, C++ Simulations) | `CUDACFD/`, `Stunticons/Source/Algebra/Solvers` (CG, BiCGSTAB in CUDA) | deterministic real-time stepping, multi-body dynamics, electronics network solves (sparse linear algebra) |
| Highly reliable autonomous flight software and the simulations that validate it; C++, Python, or Rust; fault-tolerant design; control theory | https://job-boards.greenhouse.io/spacex/jobs/8562073002 (Sr. Software Engineer, Flight Software, Starship) | `rosa/` (Rust agent), `anysignal-demo` telemetry harness (C++/Rust golden vectors) | Rust numerics library with cross-language agreement tests |

Reading of the signal for this endeavor: the simulation and GNC postings
name, verbatim, six-degree-of-freedom simulation, Monte Carlo dispersions,
post-flight correlation, real-time determinism, and Rust alongside C++.
Those five phrases set the chapter ranking in `ROADMAP.md`.
