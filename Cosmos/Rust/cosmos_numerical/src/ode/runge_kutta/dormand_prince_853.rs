//! DOP853: the Dormand–Prince explicit method of order 8 with a composite
//! 5th/3rd-order error estimator and an order-7 continuous extension,
//! HNW I §II.5 ("an 8th order method", Table 5.4 and the text after it) and
//! §II.6 (Hairer's `DOP853`, version of October 11, 2009).
//!
//! # The method
//!
//! Twelve stages give `y_1 = y_0 + h Σ b_j k_j` of order 8. A thirteenth,
//! `k_13 = f(t + h, y_1)`, is spent only once the step is accepted, and is
//! the first stage of the next step. The error is estimated from two
//! embedded combinations: `δ_5 = h Σ e_j k_j` (order-5 difference) and
//! `δ_3 = h (Σ b_j k_j − b̃_1 k_1 − b̃_9 k_9 − b̃_12 k_12)` (order-3
//! difference). With `E_5 = Σ (δ_5i/(h sc_i))²`, `E_3 = Σ (δ_3i/(h sc_i))²`,
//!
//! `err = |h| E_5 / sqrt(n (E_5 + 0.01 E_3))`,
//!
//! which behaves like an order-8 error estimate (`err ∝ h^8`) where the
//! order-5 estimate alone would overrate it at stringent tolerances, and
//! falls back to the order-5 estimate when the order-3 difference is small.
//! The step exponent is `1/8`.
//!
//! # Dense output
//!
//! Three more stages (abscissae `0.1, 0.2, 7/9`) after acceptance give
//! `r_4 … r_7 = h Σ_j d_{rj} k_j` and, with `Δ = y_1 − y_0`,
//! `β = h k_1 − Δ`, `r_0 = y_0, r_1 = Δ, r_2 = β, r_3 = Δ − h k_13 − β`,
//!
//! `u(θ) = r_0 + θ(r_1 + (1−θ)(r_2 + θ(r_3 + (1−θ) p(θ))))`,
//! `p(θ) = r_4 + θ(r_5 + (1−θ)(r_6 + θ r_7))`,
//!
//! of order 7 (`CONTD8`).
//!
//! # Stiffness
//!
//! Stages 12 and 13 share the abscissa `t + h`:
//! `h|λ| ≈ |h| ‖k_13 − k_12‖ / ‖y_1 − Y_12‖`, tested against `6.1`.

use super::embedded::{
  stage_input, stiffness_quotient, weighted_sum, EmbeddedPair,
  EmbeddedRungeKutta, IntegrationError, IntegratorOptions, Interpolant, Trial,
};
use super::tolerance::Tolerance;
use crate::field::RealField;

/// Twelve stages for the step, the thirteenth `f(t + h, y_1)`, and three for
/// the continuous extension.
pub const STAGES: usize = 16;
pub const STEP_STAGES: usize = 12;
/// The weight of the order-3 difference in the composite estimator.
pub const THIRD_ORDER_WEIGHT: f64 = 0.01;

/// The Butcher tableau, digits as in `dop853.f`. Row 13 of `A` is `b`:
/// the thirteenth stage is `f(t + h, y_1)`. `E5` and `B3` are Hairer's
/// `ER` and `BHH`; `D` holds his rows `D4`–`D7`.
pub struct Tableau;

// The literals keep every digit of dop853.f, more than f64 holds, so the
// source can be checked against the Fortran by eye.
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
impl Tableau
{
  pub const C: [f64; STAGES] = [
    0.0,
    0.526001519587677318785587544488e-01,
    0.789002279381515978178381316732e-01,
    0.118350341907227396726757197510e+00,
    0.281649658092772603273242802490e+00,
    0.333333333333333333333333333333e+00,
    0.25e+00,
    0.307692307692307692307692307692e+00,
    0.651282051282051282051282051282e+00,
    0.6e+00,
    0.857142857142857142857142857142e+00,
    1.0,
    1.0,
    0.1e+00,
    0.2e+00,
    0.777777777777777777777777777778e+00,
  ];

  pub const A: [[f64; STAGES]; STAGES] = [
    [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      5.26001519587677318785587544488e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      1.97250569845378994544595329183e-2,
      5.91751709536136983633785987549e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      2.95875854768068491816892993775e-2,
      0.0,
      8.87627564304205475450678981324e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      2.41365134159266685502369798665e-1,
      0.0,
      -8.84549479328286085344864962717e-1,
      9.24834003261792003115737966543e-1,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      3.7037037037037037037037037037e-2,
      0.0,
      0.0,
      1.70828608729473871279604482173e-1,
      1.25467687566822425016691814123e-1,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      3.7109375e-2,
      0.0,
      0.0,
      1.70252211019544039314978060272e-1,
      6.02165389804559606850219397283e-2,
      -1.7578125e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      3.70920001185047927108779319836e-2,
      0.0,
      0.0,
      1.70383925712239993810214054705e-1,
      1.07262030446373284651809199168e-1,
      -1.53194377486244017527936158236e-2,
      8.27378916381402288758473766002e-3,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      6.24110958716075717114429577812e-1,
      0.0,
      0.0,
      -3.36089262944694129406857109825e0,
      -8.68219346841726006818189891453e-1,
      2.75920996994467083049415600797e1,
      2.01540675504778934086186788979e1,
      -4.34898841810699588477366255144e1,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      4.77662536438264365890433908527e-1,
      0.0,
      0.0,
      -2.48811461997166764192642586468e0,
      -5.90290826836842996371446475743e-1,
      2.12300514481811942347288949897e1,
      1.52792336328824235832596922938e1,
      -3.32882109689848629194453265587e1,
      -2.03312017085086261358222928593e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      -9.3714243008598732571704021658e-1,
      0.0,
      0.0,
      5.18637242884406370830023853209e0,
      1.09143734899672957818500254654e0,
      -8.14978701074692612513997267357e0,
      -1.85200656599969598641566180701e1,
      2.27394870993505042818970056734e1,
      2.49360555267965238987089396762e0,
      -3.0467644718982195003823669022e0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      2.27331014751653820792359768449e0,
      0.0,
      0.0,
      -1.05344954667372501984066689879e1,
      -2.00087205822486249909675718444e0,
      -1.79589318631187989172765950534e1,
      2.79488845294199600508499808837e1,
      -2.85899827713502369474065508674e0,
      -8.87285693353062954433549289258e0,
      1.23605671757943030647266201528e1,
      6.43392746015763530355970484046e-1,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      5.42937341165687622380535766363e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      4.45031289275240888144113950566e0,
      1.89151789931450038304281599044e0,
      -5.8012039600105847814672114227e0,
      3.1116436695781989440891606237e-1,
      -1.52160949662516078556178806805e-1,
      2.01365400804030348374776537501e-1,
      4.47106157277725905176885569043e-2,
      0.0,
      0.0,
      0.0,
      0.0,
    ],
    [
      5.61675022830479523392909219681e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      2.53500210216624811088794765333e-1,
      -2.46239037470802489917441475441e-1,
      -1.24191423263816360469010140626e-1,
      1.5329179827876569731206322685e-1,
      8.20105229563468988491666602057e-3,
      7.56789766054569976138603589584e-3,
      -8.298e-3,
      0.0,
      0.0,
      0.0,
    ],
    [
      3.18346481635021405060768473261e-2,
      0.0,
      0.0,
      0.0,
      0.0,
      2.83009096723667755288322961402e-2,
      5.35419883074385676223797384372e-2,
      -5.49237485713909884646569340306e-2,
      0.0,
      0.0,
      -1.08347328697249322858509316994e-4,
      3.82571090835658412954920192323e-4,
      -3.40465008687404560802977114492e-4,
      1.41312443674632500278074618366e-1,
      0.0,
      0.0,
    ],
    [
      -4.28896301583791923408573538692e-1,
      0.0,
      0.0,
      0.0,
      0.0,
      -4.69762141536116384314449447206e0,
      7.68342119606259904184240953878e0,
      4.06898981839711007970213554331e0,
      3.56727187455281109270669543021e-1,
      0.0,
      0.0,
      0.0,
      -1.39902416515901462129418009734e-3,
      2.9475147891527723389556272149e0,
      -9.15095847217987001081870187138e0,
      0.0,
    ],
  ];

  pub const B: [f64; STAGES] = [
    5.42937341165687622380535766363e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566e0,
    1.89151789931450038304281599044e0,
    -5.8012039600105847814672114227e0,
    3.1116436695781989440891606237e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2,
    0.0,
    0.0,
    0.0,
    0.0,
  ];

  pub const E5: [f64; STAGES] = [
    0.1312004499419488073250102996e-01,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753e+01,
    -0.4957589496572501915214079952e+00,
    0.1664377182454986536961530415e+01,
    -0.3503288487499736816886487290e+00,
    0.3341791187130174790297318841e+00,
    0.8192320648511571246570742613e-01,
    -0.2235530786388629525884427845e-01,
    0.0,
    0.0,
    0.0,
    0.0,
  ];

  pub const B3: [f64; STAGES] = [
    0.244094488188976377952755905512e+00,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.733846688281611857341361741547e+00,
    0.0,
    0.0,
    0.220588235294117647058823529412e-01,
    0.0,
    0.0,
    0.0,
    0.0,
  ];

  pub const D: [[f64; STAGES]; 4] = [
    [
      -0.84289382761090128651353491142e+01,
      0.0,
      0.0,
      0.0,
      0.0,
      0.56671495351937776962531783590e+00,
      -0.30689499459498916912797304727e+01,
      0.23846676565120698287728149680e+01,
      0.21170345824450282767155149946e+01,
      -0.87139158377797299206789907490e+00,
      0.22404374302607882758541771650e+01,
      0.63157877876946881815570249290e+00,
      -0.88990336451333310820698117400e-01,
      0.18148505520854727256656404962e+02,
      -0.91946323924783554000451984436e+01,
      -0.44360363875948939664310572000e+01,
    ],
    [
      0.10427508642579134603413151009e+02,
      0.0,
      0.0,
      0.0,
      0.0,
      0.24228349177525818288430175319e+03,
      0.16520045171727028198505394887e+03,
      -0.37454675472269020279518312152e+03,
      -0.22113666853125306036270938578e+02,
      0.77334326684722638389603898808e+01,
      -0.30674084731089398182061213626e+02,
      -0.93321305264302278729567221706e+01,
      0.15697238121770843886131091075e+02,
      -0.31139403219565177677282850411e+02,
      -0.93529243588444783865713862664e+01,
      0.35816841486394083752465898540e+02,
    ],
    [
      0.19985053242002433820987653617e+02,
      0.0,
      0.0,
      0.0,
      0.0,
      -0.38703730874935176555105901742e+03,
      -0.18917813819516756882830838328e+03,
      0.52780815920542364900561016686e+03,
      -0.11573902539959630126141871134e+02,
      0.68812326946963000169666922661e+01,
      -0.10006050966910838403183860980e+01,
      0.77771377980534432092869265740e+00,
      -0.27782057523535084065932004339e+01,
      -0.60196695231264120758267380846e+02,
      0.84320405506677161018159903784e+02,
      0.11992291136182789328035130030e+02,
    ],
    [
      -0.25693933462703749003312586129e+02,
      0.0,
      0.0,
      0.0,
      0.0,
      -0.15418974869023643374053993627e+03,
      -0.23152937917604549567536039109e+03,
      0.35763911791061412378285349910e+03,
      0.93405324183624310003907691704e+02,
      -0.37458323136451633156875139351e+02,
      0.10409964950896230045147246184e+03,
      0.29840293426660503123344363579e+02,
      -0.43533456590011143754432175058e+02,
      0.96324553959188282948394950600e+02,
      -0.39177261675615439165231486172e+02,
      -0.14972683625798562581422125276e+03,
    ],
  ];
}

#[derive(Clone, Debug)]
pub struct DormandPrince853<T: RealField, const N: usize>
{
  stages: [[T; N]; STAGES],
  twelfth_stage_input: [T; N],
}

impl<T: RealField, const N: usize> Default for DormandPrince853<T, N>
{
  fn default() -> Self
  {
    Self {
      stages: [[T::zero(); N]; STAGES],
      twelfth_stage_input: [T::zero(); N],
    }
  }
}

/// `u(θ)` of one accepted DOP853 step.
#[derive(Clone, Debug)]
pub struct DormandPrince853Interpolant<T: RealField, const N: usize>
{
  coefficients: [[T; N]; 8],
}

impl<T: RealField, const N: usize> Interpolant<T, N>
  for DormandPrince853Interpolant<T, N>
{
  fn evaluate(&self, theta: T) -> [T; N]
  {
    let one_minus_theta = T::one() - theta;
    let r = &self.coefficients;
    let mut value = [T::zero(); N];
    for i in 0..N
    {
      let inner = r[4][i]
        + theta * (r[5][i] + one_minus_theta * (r[6][i] + theta * r[7][i]));
      value[i] = r[0][i]
        + theta
          * (r[1][i]
            + one_minus_theta
              * (r[2][i] + theta * (r[3][i] + one_minus_theta * inner)));
    }
    value
  }
}

impl<T: RealField, const N: usize> EmbeddedPair<T, N>
  for DormandPrince853<T, N>
{
  const NAME: &'static str = "DOP853";
  const STARTING_STEP_ORDER: u32 = 8;
  const ERROR_EXPONENT: f64 = 1.0 / 8.0;
  const BETA_CORRECTION: f64 = 0.2;
  const DEFAULT_BETA: f64 = 0.0;
  const DEFAULT_MIN_SCALE: f64 = 0.333;
  const DEFAULT_MAX_SCALE: f64 = 6.0;
  const STIFFNESS_LIMIT: f64 = 6.1;
  const EVALUATIONS_PER_ATTEMPT: usize = 11;
  const EVALUATIONS_AT_ACCEPTANCE: usize = 1;
  const EVALUATIONS_FOR_INTERPOLANT: usize = 3;

  type Interpolant = DormandPrince853Interpolant<T, N>;

  fn attempt<F>(
    &mut self,
    rhs: &mut F,
    t: T,
    y: &[T; N],
    f_0: &[T; N],
    h: T,
    tolerance: &Tolerance<T, N>) -> Trial<T, N>
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized,
  {
    self.stages[0] = *f_0;
    for stage in 1..STEP_STAGES
    {
      let input = stage_input(y, h, &Tableau::A[stage], &self.stages, stage);
      self.stages[stage] = rhs(t + T::from_f64(Tableau::C[stage]) * h, &input);
      if stage == STEP_STAGES - 1
      {
        self.twelfth_stage_input = input;
      }
    }

    let increment = weighted_sum(&Tableau::B, &self.stages, STEP_STAGES);
    let fifth_order = weighted_sum(&Tableau::E5, &self.stages, STEP_STAGES);
    let b3_1 = T::from_f64(Tableau::B3[0]);
    let b3_9 = T::from_f64(Tableau::B3[8]);
    let b3_12 = T::from_f64(Tableau::B3[11]);
    let mut y_new = [T::zero(); N];
    let mut third_order = [T::zero(); N];
    for i in 0..N
    {
      y_new[i] = y[i] + h * increment[i];
      third_order[i] = increment[i]
        - b3_1 * self.stages[0][i]
        - b3_9 * self.stages[8][i]
        - b3_12 * self.stages[11][i];
    }

    let fifth = tolerance.sum_of_squared_scaled(&fifth_order, y, &y_new);
    let third = tolerance.sum_of_squared_scaled(&third_order, y, &y_new);
    let mut denominator = fifth + T::from_f64(THIRD_ORDER_WEIGHT) * third;
    if denominator <= T::zero()
    {
      denominator = T::one();
    }
    let error = h.absolute_value()
      * fifth
      * (T::one() / (T::from_f64(N as f64) * denominator)).square_root();
    Trial { y_new, error }
  }

  fn derivative_at_end<F>(
    &mut self,
    rhs: &mut F,
    t: T,
    h: T,
    y_new: &[T; N]) -> [T; N]
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized,
  {
    self.stages[STEP_STAGES] = rhs(t + h, y_new);
    self.stages[STEP_STAGES]
  }

  fn stiffness_estimate(&self, h: T, y_new: &[T; N]) -> Option<T>
  {
    stiffness_quotient(
      h,
      &self.stages[STEP_STAGES],
      &self.stages[STEP_STAGES - 1],
      y_new,
      &self.twelfth_stage_input)
  }

  fn interpolant<F>(
    &mut self,
    rhs: &mut F,
    t: T,
    y: &[T; N],
    h: T,
    y_new: &[T; N]) -> Self::Interpolant
  where
    F: FnMut(T, &[T; N]) -> [T; N] + ?Sized,
  {
    for stage in (STEP_STAGES + 1)..STAGES
    {
      let input = stage_input(y, h, &Tableau::A[stage], &self.stages, stage);
      self.stages[stage] = rhs(t + T::from_f64(Tableau::C[stage]) * h, &input);
    }
    let mut coefficients = [[T::zero(); N]; 8];
    for row in 0..4
    {
      let sum = weighted_sum(&Tableau::D[row], &self.stages, STAGES);
      for i in 0..N
      {
        coefficients[4 + row][i] = h * sum[i];
      }
    }
    for i in 0..N
    {
      let increment = y_new[i] - y[i];
      let b_spline = h * self.stages[0][i] - increment;
      coefficients[0][i] = y[i];
      coefficients[1][i] = increment;
      coefficients[2][i] = b_spline;
      coefficients[3][i] =
        (increment - h * self.stages[STEP_STAGES][i]) - b_spline;
    }
    DormandPrince853Interpolant { coefficients }
  }
}

/// A DOP853 integration of `y' = rhs(t, y)` from `(t_0, y_0)` to `end_time`.
pub fn integrator<T, const N: usize, F>(
  rhs: F,
  t_0: T,
  y_0: [T; N],
  end_time: T,
  tolerance: Tolerance<T, N>,
  options: IntegratorOptions<T>)
  -> Result<EmbeddedRungeKutta<T, N, DormandPrince853<T, N>, F>, IntegrationError<T>>
where
  T: RealField,
  F: FnMut(T, &[T; N]) -> [T; N],
{
  EmbeddedRungeKutta::new(
    DormandPrince853::default(), rhs, t_0, y_0, end_time, tolerance, options)
}

#[cfg(test)]
mod tests
{
  use super::*;
  use crate::ode::runge_kutta::dormand_prince_5;
  use crate::ode::runge_kutta::order_conditions::{
    residuals_by_order, row_sum_defect,
  };
  use crate::ode::runge_kutta::test_problems::{
    arenstorf, ARENSTORF_INITIAL_STATE, ARENSTORF_PERIOD,
  };

  fn difference(a: &[f64; STAGES], b: &[f64; STAGES]) -> [f64; STAGES]
  {
    let mut d = [0.0; STAGES];
    for j in 0..STAGES
    {
      d[j] = a[j] - b[j];
    }
    d
  }

  /// The twelve-stage method, without the dense-output stages.
  fn step_tableau() -> [[f64; STEP_STAGES]; STEP_STAGES]
  {
    let mut a = [[0.0; STEP_STAGES]; STEP_STAGES];
    for i in 0..STEP_STAGES
    {
      a[i].copy_from_slice(&Tableau::A[i][..STEP_STAGES]);
    }
    a
  }

  fn restrict(weights: &[f64; STAGES]) -> [f64; STEP_STAGES]
  {
    let mut restricted = [0.0; STEP_STAGES];
    restricted.copy_from_slice(&weights[..STEP_STAGES]);
    restricted
  }

  #[test]
  fn the_abscissae_are_the_row_sums()
  {
    assert!(row_sum_defect(&Tableau::A, &Tableau::C) < 1.0e-14);
    assert_eq!(Tableau::A[STEP_STAGES], Tableau::B);
  }

  /// All 200 conditions of order ≤ 8 (HNW I Table 2.1 counts them), and
  /// order 9 fails.
  #[test]
  fn the_propagated_solution_has_exact_order_eight()
  {
    let residuals =
      residuals_by_order(&step_tableau(), &restrict(&Tableau::B), 9);
    assert!(residuals[..8].iter().all(|r| *r < 1.0e-13), "{residuals:?}");
    assert!(residuals[8] > 1.0e-8, "{residuals:?}");
  }

  #[test]
  fn the_error_estimators_have_orders_five_and_three()
  {
    let fifth = restrict(&difference(&Tableau::B, &Tableau::E5));
    let residuals = residuals_by_order(&step_tableau(), &fifth, 6);
    assert!(residuals[..5].iter().all(|r| *r < 1.0e-13), "{residuals:?}");
    assert!(residuals[5] > 1.0e-8, "{residuals:?}");

    let third = restrict(&Tableau::B3);
    let residuals = residuals_by_order(&step_tableau(), &third, 4);
    assert!(residuals[..3].iter().all(|r| *r < 1.0e-13), "{residuals:?}");
    assert!(residuals[3] > 1.0e-8, "{residuals:?}");
  }

  /// y' = y cos t: the step's local error falls by about 2^9 per halving,
  /// the dense output's at θ = 1/2 by about 2^8 (measured 2^8.6 and 2^8.0
  /// from h = 0.4 to 0.2; at h = 0.1 the step error reaches round-off).
  #[test]
  fn local_errors_of_the_step_and_of_the_dense_output()
  {
    let exact = |t: f64| t.sin().exp();
    let tolerance = Tolerance::uniform(1.0, 0.0).unwrap();
    let errors = |h: f64| {
      let mut rhs = |t: f64, y: &[f64; 1]| [y[0] * t.cos()];
      let mut method = DormandPrince853::<f64, 1>::default();
      let t = 0.3;
      let y = [exact(t)];
      let f_0 = rhs(t, &y);
      let trial = method.attempt(&mut rhs, t, &y, &f_0, h, &tolerance);
      method.derivative_at_end(&mut rhs, t, h, &trial.y_new);
      let interpolant = method.interpolant(&mut rhs, t, &y, h, &trial.y_new);
      (
        (trial.y_new[0] - exact(t + h)).abs(),
        (interpolant.evaluate(0.5)[0] - exact(t + 0.5 * h)).abs(),
      )
    };
    let (step_coarse, dense_coarse) = errors(0.4);
    let (step_fine, dense_fine) = errors(0.2);
    let step_order = (step_coarse / step_fine).log2() - 1.0;
    let dense_order = (dense_coarse / dense_fine).log2() - 1.0;
    assert!((step_order - 8.0).abs() < 0.5, "step order {step_order}");
    assert!((dense_order - 7.0).abs() < 0.5, "dense order {dense_order}");
  }

  #[test]
  fn closes_the_arenstorf_orbit_and_beats_dopri5_at_stringent_tolerance()
  {
    let tolerance = Tolerance::uniform(1.0e-12, 1.0e-12).unwrap();
    let mut high = integrator(
      arenstorf,
      0.0,
      ARENSTORF_INITIAL_STATE,
      ARENSTORF_PERIOD,
      tolerance,
      IntegratorOptions::default())
    .unwrap();
    let end = high.integrate_to_end().unwrap();
    let closing = (0..4)
      .map(|i| (end[i] - ARENSTORF_INITIAL_STATE[i]).abs())
      .fold(0.0, f64::max);
    assert!(closing < 1.0e-7, "{closing}");

    let mut low = dormand_prince_5::integrator(
      arenstorf,
      0.0,
      ARENSTORF_INITIAL_STATE,
      ARENSTORF_PERIOD,
      tolerance,
      IntegratorOptions::default())
    .unwrap();
    low.integrate_to_end().unwrap();
    assert!(
      high.statistics().function_evaluations
        < low.statistics().function_evaluations / 2,
      "DOP853 {:?} vs DOPRI5 {:?}",
      high.statistics(),
      low.statistics());
  }
}
