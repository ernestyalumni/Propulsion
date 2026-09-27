//! Rocket propulsion models, written from the physics and the equations of
//! Sutton, Hill & Peterson, Huzel & Huang, Humble, Turns and Williams rather
//! than from any shipped code.
//!
//! Every module names its physical objects as types, exposes every constant
//! as a validated constructor parameter, and carries property tests that
//! follow from the mathematics. Where a C++ twin exists in
//! `Cosmos/Source/Propulsion`, a golden-vector test under `golden/` proves the
//! two agree.

pub mod solid_rocket_motor;
