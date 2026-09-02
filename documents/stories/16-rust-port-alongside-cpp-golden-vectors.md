As an engineer who wants the simulation library to be mostly Rust while the existing C++ and CUDA keep working, I want every numerical module to have a Rust implementation as its primary form, with C++ and CUDA twins that are proven to agree with it, so that the languages never drift apart silently.

Every new numerical module MUST be implemented in Rust first, under the Cosmos/Rust workspace, generic over the field trait already defined in Stunticons/wildrider, and MUST pass cargo test on its own.

A C++ implementation of the same method MUST exist only when a C++ consumer in the repository needs it, and when both exist, a cross-language golden-vector test MUST exist: the C++ build emits input and output vectors to a versioned JSON file under Cosmos/Rust/golden, the Rust test reads that file, and agreement MUST be asserted to a tolerance stated in the test. A twin without a golden-vector test MUST NOT be merged.

A CUDA implementation MUST be written only for a workload that is data-parallel, and only after a CPU implementation exists and its wall time has been measured on the target problem size. Every CUDA kernel MUST be checked against the CPU implementation on the same inputs to a stated tolerance, and the measured speedup MUST be recorded next to the kernel.

Rust numerical code MUST NOT contain unsafe blocks except at a foreign-function boundary to CUDA, and that boundary MUST be isolated in its own module.

Results MUST be deterministic: the same inputs and the same seed MUST produce bitwise-identical output within one language and build configuration, and a random-number consumer MUST take its generator as an explicit parameter rather than reaching for global state.

Never let a golden-vector file be regenerated from the Rust side to make a failing comparison pass; the C++ side is the emitter and a disagreement is a finding to record.

For example, the bounded-link telemetry harness in anysignal-demo already carries a CRC-16/CCITT implementation in both C++ and Rust with a shared golden-vector test. The first numerical module to follow that pattern is the PI step-size controller: Cosmos/Source/Numerical/ODE/RKMethods/ComputePIStepSize.h emits vectors over a grid of error, previous error, step, and rejection flag, and the Rust pi_step_size module MUST reproduce every output to within 1e-15 relative.
