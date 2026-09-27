# Solid rocket motor and solid gas generator (C++)

Header-only, `template <std::floating_point Field>`. Physics and citations:
[`documents/derivations/SolidRocketMotorGasGenerator.md`](../../../../documents/derivations/SolidRocketMotorGasGenerator.md),
built on [`solid-ballistics.tex`](../../../../documents/notes/topics/solid-ballistics.tex).
Rust twin: `Cosmos/Rust/cosmos_propulsion/src/solid_rocket_motor/`. The golden
vectors in `Cosmos/Rust/golden/solid_rocket_motor_*.tsv` are emitted from these
headers.

| Header | What it is | Books |
|---|---|---|
| `CombustionProducts.h` | perfect-gas products: R, c_p, c_v, Gamma, c*, from (T_0, M, gamma) or (c*, T_0, gamma) | Turns 2.3; H&P 3.14; Sutton 3-32 |
| `BurningRate.h` | r = a pⁿ; exponential (Sutton) and explosion-temperature (Williams 7-41) grain-temperature laws; lumped Lenoir–Robillard erosive burning | Sutton 12-5, 12-12, 12-17; Williams §7 |
| `Grain.h` | tubular (inhibited or burning ends) and end-burning grains; dV/dy = A_b by construction | Sutton §12.3 |
| `CompressibleFlow.h` | choked / subsonic / reversed restriction flow; supersonic exit Mach; ideal thrust | Sutton 3-16, 3-24, 3-25, 3-30 |
| `PintleValve.h` | conical pintle A(x) = πR²[1 − (1 − x)²], piecewise-linear opening schedules | Sutton p. 328, Fig. 12-27 |
| `RungeKutta4.h` | named RK4 tableau; a step that lands exactly on burnout | — |
| `GrainChamber.h` | lumped chamber on (y, m_c) | Sutton 12-1, 12-3; H&P 12.28; Humble 6.36 |
| `SolidRocketMotor.h` | chamber → nozzle → ambient | |
| `GasGeneratorValveSystem.h` | chamber → orifice → plenum → N pintle valves → ambient | Turns 6.28/6.34; Huzel pp. 116, 148–149 |
| `Scenarios.h` | Huzel & Huang p. 116 cartridge; illustrative tubular booster; cartridge + 4 valves | |

Tests: `Cosmos/Source/UnitTests/Propulsion/SolidRocketMotor/` (built into `Check`).

```bash
cd Cosmos/BuildGcc && cmake ../Source -DCMAKE_BUILD_TYPE=Release && make -j8 Check
./Check --gtest_filter='*SolidRocketMotor*:*GasGenerator*:*BurningRate*:*Grain*:*CompressibleFlow*:*PintleValve*:*CombustionProducts*'
```
