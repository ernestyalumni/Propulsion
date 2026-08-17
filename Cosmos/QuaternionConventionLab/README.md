# Quaternion Convention Lab

An interview-sized spacecraft simulation artifact that answers two questions precisely:

1. Why do `q` and `−q` encode the same physical attitude?
2. Why can two quaternion libraries disagree even when both inputs are normalized?

Those are different phenomena. Unit quaternions (`SU(2)`) double-cover `SO(3)`, which explains the antipodal equivalence `R(q) = R(−q)`. Scalar layout, multiplication definition, active/passive action, frame direction, and composition order are engineering conventions. The topology does not choose those conventions.

## The contract in this lab

- Hamilton multiplication (`i*j = k`)
- scalar-first C++/Rust API: `(w, x, y, z)`
- active vector rotation
- body/local frame to inertial/world frame
- `v_world = q * (0, v_body) * conjugate(q)`
- `q2*q1` applies `q1` and then `q2`

The phrase “JPL convention” is not sufficient at an API boundary. JPL’s NAIF SPICE documentation, for example, uses Hamilton multiplication and scalar-first layout while also documenting a different popular scalar-last style. Name every relevant choice and provide an explicit adapter.

## What is here

- `cpp/`: dependency-free C++20 kernel and executable assertions
- `rust/`: matching dependency-free Rust kernel and unit tests
- `web/`: interactive Three.js spacecraft visualization with two failure injections
- `slides/slides.html`: a standalone two-slide briefing
- `VIDEO_SCRIPT.md`: five-minute YouTube script, short-form script, and recording plan

## Run the browser demo

The demo reuses the repository’s vendored Three.js, so no install or internet connection is required.

```bash
cd Cosmos
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000/QuaternionConventionLab/web/`.

Use **q versus −q** to show zero physical error through a 720° path. Use **Convention mismatch** to inject either a scalar-layout error or an active/passive inversion.

For a repeatable recording frame, deep-link to a failure, for example: `?mode=mismatch&mismatch=layout&angle=90`.

Open the slides at `http://127.0.0.1:8000/QuaternionConventionLab/slides/slides.html`; navigate with the arrow keys. Add `?slide=2` to open directly on slide two.

## Build and test without polluting the repository

### C++20

```bash
cmake -S Cosmos/QuaternionConventionLab/cpp \
      -B /tmp/quaternion-convention-cpp \
      -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quaternion-convention-cpp
/tmp/quaternion-convention-cpp/quaternion_convention
```

### Rust

```bash
CARGO_TARGET_DIR=/tmp/quaternion-convention-rust \
  cargo test --manifest-path Cosmos/QuaternionConventionLab/rust/Cargo.toml
CARGO_TARGET_DIR=/tmp/quaternion-convention-rust \
  cargo run --manifest-path Cosmos/QuaternionConventionLab/rust/Cargo.toml
```

## The 30-second interview answer

“There are two separate issues. Mathematically, `q` and `−q` encode the same attitude because `SU(2)` is a two-to-one cover of `SO(3)`. That means attitude metrics and interpolation must handle antipodal equivalence. Separately, libraries choose conventions—scalar order, multiplication definition, active versus passive action, frame direction, and composition order. I would declare the full contract, adapt only at boundaries, and lock it down with a known test such as positive 90 degrees about positive Z mapping positive X to positive Y.”

## Verification matrix

| Invariant / failure | C++ | Rust | Browser |
|---|:---:|:---:|:---:|
| `R(q) = R(−q)` | yes | yes | live |
| sign-invariant physical error | yes | yes | live |
| +90° about +Z maps +X to +Y | yes | yes | visual axes |
| scalar-first/scalar-last adapter | yes | yes | injected mismatch |
| active/passive distinction | yes | kernel supports inverse | injected mismatch |
| hemisphere alignment | yes | yes | 720° path |

## Scope

This is deliberately not a full ADCS or 6DOF simulator. It is a small, auditable boundary-contract demo. The companion True Anomaly drill in `repos/Galvatron/Documents/TrueAnomaly/drill_6dof_and_events.cpp` covers 13-state rigid-body propagation, gravity-gradient torque, RK4 convergence, event localization, J2, and rotating-atmosphere drag.

## References

- [JPL NAIF SPICE quaternion styles](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/IDL/icy/cspice_qdq2av.html)
- [NASA attitude mathematics, NASA/TP–2018–219822](https://ntrs.nasa.gov/api/citations/20180003657/downloads/20180003657.pdf)
- [Quaternions and Attitude Representation](https://arxiv.org/abs/1708.08680)
- [SU(2) as a double cover of SO(3), University of Alberta](https://sites.ualberta.ca/~vbouchar/MAPH464/section-su2.html)
