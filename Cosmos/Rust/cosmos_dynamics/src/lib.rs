//! Dynamics for spacecraft simulation and GNC: the equations of motion, their
//! invariants, and the integrators that respect them.
//!
//! The ODE machinery (Runge–Kutta steps, embedded pairs, step-size control)
//! lives in `cosmos_numerical`; this crate supplies right-hand sides shaped
//! for it, `FnMut(f64, &[f64; N]) -> [f64; N]`, plus structure-preserving
//! steps that a general-purpose integrator cannot provide.

// Index loops are kept where they mirror subscripted equations (J_ij, L_ik)
// and the summation order of the golden twins; iterator rewrites hide both.
#![allow(clippy::needless_range_loop)]

pub mod rigid_body;
