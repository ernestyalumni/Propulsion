//! The WebAssembly boundary between the Cosmos integrators and a TypeScript
//! viewer.
//!
//! Rust owns the simulation; TypeScript owns the clock, the input and the
//! rendering. Each frame the viewer calls `advance_to(t)` once and then reads
//! the results without copying, through `Float64Array` views over this
//! module's linear memory at the addresses `frame_pointer()`,
//! `trail_pointer()` and `markers_pointer()` return. The buffers are
//! allocated once and never reallocate, but a view must still be rebuilt
//! whenever `memory.buffer` changes (a memory growth detaches it).
//!
//! The labs themselves (`rigid_body_lab`, `orbit_lab`) are plain Rust with
//! native tests; this file only adapts them to `wasm-bindgen`.

// Index loops mirror the subscripted equations.
#![allow(clippy::needless_range_loop)]

pub mod orbit_lab;
pub mod rigid_body_lab;

use wasm_bindgen::prelude::*;

/// Four integrators on one rigid body, plus a DOP853 reference.
#[wasm_bindgen(js_name = RigidBodyLab)]
pub struct WebRigidBodyLab
{
  lab: rigid_body_lab::RigidBodyLab,
}

#[wasm_bindgen(js_class = RigidBodyLab)]
impl WebRigidBodyLab
{
  /// `scenario`: 0 intermediate-axis flip, 1 precession, 2 tumble.
  #[wasm_bindgen(constructor)]
  pub fn new(
    scenario: u32,
    time_step: f64,
    tolerance: f64) -> Result<WebRigidBodyLab, JsError>
  {
    let scenario = rigid_body_lab::Scenario::from_index(scenario)
      .ok_or_else(|| JsError::new("unknown scenario"))?;
    rigid_body_lab::RigidBodyLab::new(scenario, time_step, tolerance)
      .map(|lab| Self { lab })
      .map_err(|error| JsError::new(&error.to_string()))
  }

  #[wasm_bindgen(js_name = advanceTo)]
  pub fn advance_to(&mut self, time: f64)
  {
    self.lab.advance_to(time);
  }

  #[wasm_bindgen(js_name = setTorque)]
  pub fn set_torque(&mut self, x: f64, y: f64, z: f64)
  {
    self.lab.set_torque([x, y, z]);
  }

  #[wasm_bindgen(js_name = framePointer)]
  pub fn frame_pointer(&self) -> *const f64
  {
    self.lab.frame().as_ptr()
  }

  #[wasm_bindgen(js_name = frameLength)]
  pub fn frame_length(&self) -> usize
  {
    self.lab.frame().len()
  }

  #[wasm_bindgen(js_name = frameStride)]
  pub fn frame_stride() -> usize
  {
    rigid_body_lab::FRAME_STRIDE
  }

  #[wasm_bindgen(js_name = bodyCount)]
  pub fn body_count() -> usize
  {
    rigid_body_lab::BODY_COUNT
  }

  #[wasm_bindgen(js_name = methodName)]
  pub fn method_name(index: usize) -> String
  {
    rigid_body_lab::METHODS
      .get(index)
      .map_or("reference DOP853", |method| method.name())
      .to_string()
  }
}

/// The Arenstorf orbit under DOPRI5 and DOP853.
#[wasm_bindgen(js_name = OrbitLab)]
pub struct WebOrbitLab
{
  lab: orbit_lab::OrbitLab,
}

#[wasm_bindgen(js_class = OrbitLab)]
impl WebOrbitLab
{
  #[wasm_bindgen(constructor)]
  pub fn new(tolerance: f64) -> Result<WebOrbitLab, JsError>
  {
    orbit_lab::OrbitLab::new(tolerance)
      .map(|lab| Self { lab })
      .map_err(|error| JsError::new(&error.to_string()))
  }

  #[wasm_bindgen(js_name = advanceTo)]
  pub fn advance_to(&mut self, time: f64)
  {
    self.lab.advance_to(time);
  }

  #[wasm_bindgen(js_name = framePointer)]
  pub fn frame_pointer(&self) -> *const f64
  {
    self.lab.frame().as_ptr()
  }

  #[wasm_bindgen(js_name = frameStride)]
  pub fn frame_stride() -> usize
  {
    orbit_lab::FRAME_STRIDE
  }

  #[wasm_bindgen(js_name = trailPointer)]
  pub fn trail_pointer(&self, method: usize) -> *const f64
  {
    self.lab.trail(method).as_ptr()
  }

  #[wasm_bindgen(js_name = trailLength)]
  pub fn trail_length(&self, method: usize) -> usize
  {
    self.lab.trail(method).len()
  }

  #[wasm_bindgen(js_name = markersPointer)]
  pub fn markers_pointer(&self, method: usize) -> *const f64
  {
    self.lab.markers(method).as_ptr()
  }

  #[wasm_bindgen(js_name = markersLength)]
  pub fn markers_length(&self, method: usize) -> usize
  {
    self.lab.markers(method).len()
  }

  #[wasm_bindgen(js_name = period)]
  pub fn period() -> f64
  {
    orbit_lab::PERIOD
  }

  #[wasm_bindgen(js_name = mu)]
  pub fn mu() -> f64
  {
    orbit_lab::MU
  }
}
