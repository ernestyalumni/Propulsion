# Cosmos/Rust — the Rust side of the Cosmos numerical library

Rust is the primary language for new numerical modules (story 16). This
workspace mirrors `Cosmos/Source` module for module; `cosmos_numerical` is the
twin of `Cosmos/Source/Numerical`, and `cosmos_propulsion` is the twin of
`Cosmos/Source/Propulsion`.

```bash
cd Cosmos/Rust
cargo test            # every module, including the golden-vector comparisons
cargo doc --open      # derivations are linked from each module's docs
```

## Golden vectors

When a method exists in both C++ and Rust, the C++ build is the emitter and the
Rust test is the reader. Files live in `golden/` and are regenerated only from
the C++ tool in `tools/`; never edit them by hand and never regenerate them from
the Rust side to make a comparison pass.

```bash
# from the repository root
g++ -std=c++17 -O2 -I Cosmos/Source Cosmos/Rust/tools/emit_pi_step_size_golden.cpp -o /tmp/emit
/tmp/emit     > Cosmos/Rust/golden/pi_step_size.json   # for humans
/tmp/emit tsv > Cosmos/Rust/golden/pi_step_size.tsv    # what the Rust test reads
```

## Style

Two-space indent, opening brace on its own line, spelled-out names, snake_case
functions, CamelCase types, one mathematical noun per type, every constant a
named constructor parameter (story 15). `rustfmt.toml` pins what rustfmt can
pin; brace placement is kept by hand.

## Modules

| Module | NR section | Derivation | C++ twin | Golden vectors |
|---|---|---|---|---|
| `ode::runge_kutta::pi_step_size` | 17.2 | `ComputePIStepSize.h` comments; HNW I §II.4 | `Numerical/ODE/RKMethods/ComputePIStepSize.h` | `golden/pi_step_size.tsv` |
| `ode::runge_kutta::{tolerance, initial_step, embedded, dormand_prince_5, dormand_prince_853}` (error norm, starting step, adaptive driver with dense output and stiffness test, DOPRI5, DOP853) | 17.2 | module docs; HNW I §II.4 eqs. (4.10)–(4.14), §II.5, §II.6; order conditions checked on all rooted trees (`order_conditions`, §II.2) | `Numerical/ODE/RKMethods/` (DOPRI5); Hairer's Fortran `dopri5.f`, `dop853.f` | `golden/hairer_reference/*.tsv` (bitwise identical to Hairer's codes on x86-64 Linux) |
| `ode::runge_kutta::classical_4` | 17.1 | HNW I §II.1 Table 1.2; re-exported by `cosmos_propulsion::solid_rocket_motor::runge_kutta_4` | `Propulsion/SolidRocketMotor/RungeKutta4.h` | via `golden/solid_rocket_motor_*.tsv` (bitwise) |
| `cosmos_dynamics::rigid_body` (6DOF state/model, `derivative_array` for integrators, invariants, quaternion projection, `ImplicitMidpoint`, RK4 propagation) | — (GNI §IV.2, §IV.4, §VII.5; HNW I §II.7) | module docs; `mathphysics/Rust/lie-groups/src/rigid_body.rs` | mathphysics `lie-groups::rigid_body` (Rust) | `golden/rigid_body_mathphysics_fixtures.tsv` (bitwise) |
| `linear_algebra::cholesky` | 2.9 | `documents/derivations/CholeskyFactorization.md` | none yet | none yet |
| `cosmos_propulsion::solid_rocket_motor` (grain, burning rate, erosive burning, compressible flow, pintle valves, motor, gas generator + plenum + N valves) | — (Sutton ch. 12, H&P §12.7, Humble §6.5, Huzel pp. 116/148, Turns §6, Williams ch. 7) | `documents/derivations/SolidRocketMotorGasGenerator.md` | `Propulsion/SolidRocketMotor/*.h` | `golden/solid_rocket_motor_{components,huzel_cartridge,tubular_booster,gas_generator}.tsv` (trajectories bitwise identical to C++ on x86-64 Linux) |

### Hairer reference golden vectors

Hairer's `dopri5.f` and `dop853.f` are fetched (hash-checked, not vendored),
compiled with gfortran, and run on the Arenstorf orbit and Prothero–Robinson;
see `tools/hairer_reference/README.md`.

```bash
Cosmos/Rust/tools/hairer_reference/emit_golden.sh
cargo test -p cosmos_numerical hairer_reference
```

### Rigid-body golden vectors

Emitted by the mathphysics twin; the provenance header in the TSV names the
commit. Regenerate (outside this repository's target directory) and convert
the JSON lines to the TSV columns:

```bash
cd mathphysics/Rust/lie-groups
CARGO_TARGET_DIR=/tmp/lie-groups-target cargo run --release --example rigid_body_fixtures
```

### Solid rocket motor golden vectors

```bash
# from the repository root
g++ -std=c++20 -O2 -I Cosmos/Source Cosmos/Rust/tools/emit_solid_rocket_motor_golden.cpp -o /tmp/emit_srm
for table in components huzel_cartridge tubular_booster gas_generator; do
  /tmp/emit_srm $table > Cosmos/Rust/golden/solid_rocket_motor_$table.tsv
done
# run a scenario from Rust: huzel_cartridge | tubular_booster | gas_generator
cargo run --release -p cosmos_propulsion --example solid_rocket_motor -- gas_generator
```

