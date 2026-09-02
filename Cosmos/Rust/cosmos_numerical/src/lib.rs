//! Numerical methods for spacecraft simulation and GNC, written from the
//! physics and the equations rather than from any textbook's shipped code.
//!
//! Every module names the mathematical objects it computes with (a tableau, a
//! factor, a controller law) as types, exposes every constant as an asserted
//! constructor parameter, and carries property tests that follow from the
//! mathematics. Where a C++ twin exists in `Cosmos/Source`, a golden-vector
//! test under `golden/` proves the two agree.

pub mod field;
pub mod linear_algebra;
pub mod ode;
