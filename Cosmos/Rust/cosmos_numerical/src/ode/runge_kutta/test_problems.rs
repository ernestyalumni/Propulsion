//! Test problems shared by the method tests and the golden-vector tests.
//! Constants carry HNW's digits.
#![allow(clippy::excessive_precision)]

/// μ of the Arenstorf orbit, HNW I eq. (0.1).
pub const ARENSTORF_MU: f64 = 0.012277471;
pub const ARENSTORF_INITIAL_STATE: [f64; 4] =
  [0.994, 0.0, 0.0, -2.00158510637908252240537862224];
pub const ARENSTORF_PERIOD: f64 = 17.0652165601579625588917206249;

/// The restricted three-body problem of HNW I eq. (0.1), as a first-order
/// system `(y_1, y_2, y_1', y_2')`. `D^{3/2}` is `D √D`, as in Hairer's own
/// drivers and in `tools/hairer_reference/reference_driver.F90`: the square
/// root is correctly rounded everywhere, `powf` is not.
pub fn arenstorf(_t: f64, y: &[f64; 4]) -> [f64; 4]
{
  let mu = ARENSTORF_MU;
  let mu_prime = 1.0 - mu;
  let r_1 = (y[0] + mu) * (y[0] + mu) + y[1] * y[1];
  let d_1 = r_1 * r_1.sqrt();
  let r_2 = (y[0] - mu_prime) * (y[0] - mu_prime) + y[1] * y[1];
  let d_2 = r_2 * r_2.sqrt();
  [
    y[2],
    y[3],
    y[0] + 2.0 * y[3] - mu_prime * (y[0] + mu) / d_1 - mu * (y[0] - mu_prime) / d_2,
    y[1] - 2.0 * y[2] - mu_prime * y[1] / d_1 - mu * y[1] / d_2,
  ]
}

/// Prothero–Robinson, `y' = −1000 (y − cos t) − sin t`, `y = cos t`: smooth
/// solution, stiff Jacobian.
pub fn prothero_robinson(t: f64, y: &[f64; 1]) -> [f64; 1]
{
  [-1000.0 * (y[0] - t.cos()) - t.sin()]
}
