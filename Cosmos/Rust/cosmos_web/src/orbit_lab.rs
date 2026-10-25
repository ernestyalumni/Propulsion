//! The Arenstorf orbit of HNW I §II.0, eq. (0.1), integrated by DOPRI5 and
//! DOP853 at one tolerance, recording every accepted step.
//!
//! The orbit is periodic with period `T ≈ 17.0652`; the distance between the
//! state at each multiple of `T` and the initial state is the global error
//! the tolerance actually bought. The accepted steps cluster where the
//! satellite passes the moon, which is what step-size control is for.

use cosmos_numerical::ode::runge_kutta::dormand_prince_5::DormandPrince5;
use cosmos_numerical::ode::runge_kutta::dormand_prince_853::DormandPrince853;
use cosmos_numerical::ode::runge_kutta::embedded::{
  EmbeddedPair, EmbeddedRungeKutta, IntegratorOptions,
};
use cosmos_numerical::ode::runge_kutta::tolerance::Tolerance;

use crate::rigid_body_lab::LabError;

pub const MU: f64 = 0.012277471;
#[allow(clippy::excessive_precision)]
pub const INITIAL_STATE: [f64; 4] =
  [0.994, 0.0, 0.0, -2.00158510637908252240537862224];
#[allow(clippy::excessive_precision)]
pub const PERIOD: f64 = 17.0652165601579625588917206249;
pub const METHOD_COUNT: usize = 2;
pub const METHOD_NAMES: [&str; METHOD_COUNT] = ["DOPRI5", "DOP853"];
pub const FRAME_STRIDE: usize = 12;
/// `(x, y)` pairs kept for the trail drawn behind each satellite.
pub const TRAIL_CAPACITY: usize = 8192;
/// `(x, y, h)` triples kept for the accepted-step markers.
pub const MARKER_CAPACITY: usize = 8192;
const HORIZON: f64 = 1.0e6;

pub mod frame
{
  pub const TIME: usize = 0;
  pub const POSITION: usize = 1;
  pub const VELOCITY: usize = 3;
  pub const ACCEPTED: usize = 5;
  pub const REJECTED: usize = 6;
  pub const EVALUATIONS: usize = 7;
  pub const STEP_SIZE: usize = 8;
  pub const FAILED: usize = 9;
  pub const CLOSING_ERROR: usize = 10;
  pub const PERIODS: usize = 11;
}

/// The restricted three-body right-hand side in the rotating frame.
pub fn arenstorf(_t: f64, y: &[f64; 4]) -> [f64; 4]
{
  let mu_prime = 1.0 - MU;
  let r_1 = (y[0] + MU) * (y[0] + MU) + y[1] * y[1];
  let d_1 = r_1 * r_1.sqrt();
  let r_2 = (y[0] - mu_prime) * (y[0] - mu_prime) + y[1] * y[1];
  let d_2 = r_2 * r_2.sqrt();
  [
    y[2],
    y[3],
    y[0] + 2.0 * y[3] - mu_prime * (y[0] + MU) / d_1 - MU * (y[0] - mu_prime) / d_2,
    y[1] - 2.0 * y[2] - mu_prime * y[1] / d_1 - MU * y[1] / d_2,
  ]
}

trait OrbitRun
{
  /// Steps past `time`, appending `(x, y, h)` for each accepted step, and
  /// returns the solution at `time`.
  fn advance(&mut self, time: f64, markers: &mut Vec<f64>) -> Result<[f64; 4], String>;
  fn counts(&self) -> (usize, usize, usize);
  fn last_step_size(&self) -> f64;
}

impl<M, F> OrbitRun for EmbeddedRungeKutta<f64, 4, M, F>
where
  M: EmbeddedPair<f64, 4>,
  F: FnMut(f64, &[f64; 4]) -> [f64; 4],
{
  fn advance(&mut self, time: f64, markers: &mut Vec<f64>) -> Result<[f64; 4], String>
  {
    if time > self.end_time()
    {
      self.extend_end_time(time + HORIZON).map_err(|error| error.to_string())?;
    }
    while time > self.time()
    {
      let step = self.step().map_err(|error| error.to_string())?;
      if let Some(step) = step
      {
        let state = self.state();
        push_bounded(markers, &[state[0], state[1], step.step_size], MARKER_CAPACITY * 3);
      }
    }
    if time == self.time()
    {
      return Ok(*self.state());
    }
    self.dense_output(time).map_err(|error| error.to_string())
  }

  fn counts(&self) -> (usize, usize, usize)
  {
    let statistics = self.statistics();
    (statistics.accepted, statistics.rejected, statistics.function_evaluations)
  }

  fn last_step_size(&self) -> f64
  {
    self
      .last_step_interval()
      .map_or(self.step_size(), |(start, end)| end - start)
  }
}

/// Appends `values`, dropping the oldest half when `capacity` would be
/// exceeded. The buffer never reallocates, so a view of it from JavaScript
/// stays at the same address.
fn push_bounded(buffer: &mut Vec<f64>, values: &[f64], capacity: usize)
{
  if buffer.len() + values.len() > capacity
  {
    let keep_from = (buffer.len() / 2) / values.len() * values.len();
    buffer.drain(..keep_from);
  }
  buffer.extend_from_slice(values);
}

struct Track
{
  run: Box<dyn OrbitRun>,
  trail: Vec<f64>,
  markers: Vec<f64>,
  next_period: f64,
  periods: usize,
  closing_error: f64,
  failed: bool,
}

pub struct OrbitLab
{
  tracks: Vec<Track>,
  frame: Vec<f64>,
}

fn new_run<M>(tolerance: f64) -> Result<Box<dyn OrbitRun>, LabError>
where
  M: EmbeddedPair<f64, 4> + Default + 'static,
{
  let tolerance = Tolerance::uniform(tolerance, tolerance)
    .map_err(|_| LabError::InvalidSetting("tolerance must be positive"))?;
  let options = IntegratorOptions {
    maximum_attempts: usize::MAX,
    ..IntegratorOptions::default()
  };
  let run = EmbeddedRungeKutta::new(
    M::default(), arenstorf, 0.0, INITIAL_STATE, HORIZON, tolerance, options)
    .map_err(|error| LabError::InvalidModel(error.to_string()))?;
  Ok(Box::new(run))
}

impl OrbitLab
{
  pub fn new(tolerance: f64) -> Result<Self, LabError>
  {
    let runs = [
      new_run::<DormandPrince5<f64, 4>>(tolerance)?,
      new_run::<DormandPrince853<f64, 4>>(tolerance)?,
    ];
    let tracks = runs
      .into_iter()
      .map(|run| Track {
        run,
        trail: Vec::with_capacity(TRAIL_CAPACITY * 2),
        markers: Vec::with_capacity(MARKER_CAPACITY * 3),
        next_period: PERIOD,
        periods: 0,
        closing_error: f64::NAN,
        failed: false,
      })
      .collect();
    let mut lab = Self { tracks, frame: vec![0.0; METHOD_COUNT * FRAME_STRIDE] };
    lab.advance_to(0.0);
    Ok(lab)
  }

  pub fn frame(&self) -> &[f64] { &self.frame }
  pub fn trail(&self, method: usize) -> &[f64] { &self.tracks[method].trail }
  pub fn markers(&self, method: usize) -> &[f64] { &self.tracks[method].markers }

  pub fn advance_to(&mut self, time: f64)
  {
    for (index, track) in self.tracks.iter_mut().enumerate()
    {
      let mut values = [0.0; FRAME_STRIDE];
      values[frame::TIME] = time;
      if !track.failed
      {
        let mut outcome = Ok(INITIAL_STATE);
        while track.next_period <= time && outcome.is_ok()
        {
          outcome = track.run.advance(track.next_period, &mut track.markers);
          if let Ok(state) = outcome
          {
            track.closing_error = (0..4)
              .map(|i| (state[i] - INITIAL_STATE[i]).abs())
              .fold(0.0, f64::max);
            track.periods += 1;
            track.next_period = (track.periods + 1) as f64 * PERIOD;
          }
        }
        if outcome.is_ok()
        {
          outcome = track.run.advance(time, &mut track.markers);
        }
        match outcome
        {
          Ok(state) =>
          {
            push_bounded(&mut track.trail, &state[..2], TRAIL_CAPACITY * 2);
            values[frame::POSITION..frame::POSITION + 4].copy_from_slice(&state);
          }
          Err(_) => track.failed = true,
        }
      }
      let (accepted, rejected, evaluations) = track.run.counts();
      values[frame::ACCEPTED] = accepted as f64;
      values[frame::REJECTED] = rejected as f64;
      values[frame::EVALUATIONS] = evaluations as f64;
      values[frame::STEP_SIZE] = track.run.last_step_size();
      values[frame::FAILED] = if track.failed { 1.0 } else { 0.0 };
      values[frame::CLOSING_ERROR] = track.closing_error;
      values[frame::PERIODS] = track.periods as f64;
      let base = index * FRAME_STRIDE;
      self.frame[base..base + FRAME_STRIDE].copy_from_slice(&values);
    }
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn one_period_at_sixty_frames_per_second()
  {
    let mut lab = OrbitLab::new(1.0e-9).unwrap();
    let frames = (PERIOD * 60.0).ceil() as usize + 1;
    for frame_number in 1..=frames
    {
      lab.advance_to(frame_number as f64 / 60.0);
    }
    for method in 0..METHOD_COUNT
    {
      let values = &lab.frame()[method * FRAME_STRIDE..(method + 1) * FRAME_STRIDE];
      assert_eq!(values[frame::FAILED], 0.0);
      assert_eq!(values[frame::PERIODS], 1.0);
      assert!(values[frame::CLOSING_ERROR] < 1.0e-4, "{values:?}");
      assert_eq!(lab.markers(method).len() / 3, values[frame::ACCEPTED] as usize);
      assert_eq!(lab.trail(method).len(), 2 * (frames + 1));
    }
    let dopri5 = lab.frame()[frame::EVALUATIONS];
    let dop853 = lab.frame()[FRAME_STRIDE + frame::EVALUATIONS];
    assert!(dop853 < dopri5, "DOP853 {dop853} vs DOPRI5 {dopri5}");
  }

  #[test]
  fn bounded_buffers_never_reallocate()
  {
    let mut buffer: Vec<f64> = Vec::with_capacity(12);
    let address = buffer.as_ptr();
    for i in 0..100
    {
      push_bounded(&mut buffer, &[i as f64, 0.0, 1.0], 12);
      assert!(buffer.len() <= 12 && buffer.len().is_multiple_of(3));
    }
    assert_eq!(buffer.as_ptr(), address);
    assert_eq!(buffer[buffer.len() - 3], 99.0);
  }
}
