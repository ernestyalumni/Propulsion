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
| `linear_algebra::cholesky` | 2.9 | `documents/derivations/CholeskyFactorization.md` | none yet | none yet |
| `cosmos_propulsion::solid_rocket_motor` (grain, burning rate, erosive burning, compressible flow, pintle valves, motor, gas generator + plenum + N valves) | — (Sutton ch. 12, H&P §12.7, Humble §6.5, Huzel pp. 116/148, Turns §6, Williams ch. 7) | `documents/derivations/SolidRocketMotorGasGenerator.md` | `Propulsion/SolidRocketMotor/*.h` | `golden/solid_rocket_motor_{components,huzel_cartridge,tubular_booster,gas_generator}.tsv` (trajectories bitwise identical to C++ on x86-64 Linux) |

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

