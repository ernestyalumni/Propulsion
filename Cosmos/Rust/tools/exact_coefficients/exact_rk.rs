//! Exact RK output-weight derivation for a fixed explicit stage matrix.
//! HNW I, II.1 and II.2: b.Phi(t)=1/gamma(t), c=A*1.
//! Standalone: rustc --edition=2021 exact_rk.rs -o /tmp/exact-rk
//! All integer operations are checked: oversized calculations panic, never wrap.
use std::{
    fmt,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    n: i128,
    d: i128,
}
impl Rational {
    pub fn new(n: i128, d: i128) -> Self {
        assert!(d != 0, "zero denominator");
        let (n, d) = if d < 0 {
            (n.checked_neg().unwrap(), d.checked_neg().unwrap())
        } else {
            (n, d)
        };
        let (mut a, mut b) = (n.checked_abs().unwrap(), d);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Self { n: n / a, d: d / a }
    }
}
impl From<i128> for Rational {
    fn from(n: i128) -> Self {
        Self::new(n, 1)
    }
}
impl Add for Rational {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(
            self.n
                .checked_mul(r.d)
                .unwrap()
                .checked_add(r.n.checked_mul(self.d).unwrap())
                .unwrap(),
            self.d.checked_mul(r.d).unwrap(),
        )
    }
}
impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(self.n.checked_neg().unwrap(), self.d)
    }
}
impl Sub for Rational {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        self + -r
    }
}
impl Mul for Rational {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.n.checked_mul(r.n).unwrap(),
            self.d.checked_mul(r.d).unwrap(),
        )
    }
}
impl Div for Rational {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        self * Self::new(r.d, r.n)
    }
}
impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 {
            write!(f, "{}", self.n)
        } else {
            write!(f, "{}/{}", self.n, self.d)
        }
    }
}
fn q(n: i128, d: i128) -> Rational {
    Rational::new(n, d)
}
fn zero() -> Rational {
    0.into()
}
fn one() -> Rational {
    1.into()
}
fn dot(a: &[Rational], b: &[Rational]) -> Rational {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(zero(), |s, (&x, &y)| s + x * y)
}

#[derive(Debug)]
struct Tree {
    order: usize,
    children: Vec<usize>,
}
fn trees(p: usize) -> Vec<Tree> {
    fn choose(
        ts: &[Tree],
        left: usize,
        start: usize,
        children: &mut Vec<usize>,
        out: &mut Vec<Tree>,
        order: usize,
    ) {
        if left == 0 {
            out.push(Tree {
                order,
                children: children.clone(),
            });
            return;
        }
        for i in start..ts.len() {
            if ts[i].order <= left {
                children.push(i);
                choose(ts, left - ts[i].order, i, children, out, order);
                children.pop();
            }
        }
    }
    let mut ts = vec![];
    for order in 1..=p {
        let mut next = vec![];
        choose(&ts, order - 1, 0, &mut vec![], &mut next, order);
        ts.extend(next);
    }
    ts
}
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidMatrix,
    UnsupportedOrder,
    Inconsistent,
    Underdetermined,
}
#[derive(Debug)]
pub struct Condition {
    pub order: usize,
    pub weights: Vec<Rational>,
    pub target: Rational,
}
/// Generate every rooted-tree condition; restrict order to 1..=8 to bound enumeration.
/// Nonautonomous consistency uses c=A*1, as returned by `abscissae`.
pub fn conditions(a: &[Vec<Rational>], p: usize) -> Result<Vec<Condition>, Error> {
    if !(1..=8).contains(&p) {
        return Err(Error::UnsupportedOrder);
    }
    let s = a.len();
    if s == 0
        || a.iter()
            .enumerate()
            .any(|(i, row)| row.len() != s || row[i..].iter().any(|&v| v != zero()))
    {
        return Err(Error::InvalidMatrix);
    }
    let mut result: Vec<Condition> = vec![];
    let mut gamma = vec![];
    for t in trees(p) {
        let mut phi = vec![one(); s];
        let mut density = Rational::from(t.order as i128);
        for child in t.children {
            for i in 0..s {
                phi[i] = phi[i] * dot(&a[i], &result[child].weights);
            }
            density = density * gamma[child];
        }
        result.push(Condition {
            order: t.order,
            weights: phi,
            target: one() / density,
        });
        gamma.push(density);
    }
    Ok(result)
}
pub fn abscissae(a: &[Vec<Rational>]) -> Vec<Rational> {
    a.iter()
        .map(|r| r.iter().copied().fold(zero(), |x, y| x + y))
        .collect()
}
/// Solve ALL order conditions simultaneously by exact Gaussian elimination.
/// Rejects inconsistent and nonunique systems rather than choosing free weights.
pub fn derive_weights(a: &[Vec<Rational>], p: usize) -> Result<Vec<Rational>, Error> {
    let cs = conditions(a, p)?;
    let s = a.len();
    let mut rows: Vec<Vec<Rational>> = cs
        .iter()
        .map(|c| {
            let mut r = c.weights.clone();
            r.push(c.target);
            r
        })
        .collect();
    let mut pivots = vec![];
    for col in 0..s {
        let rank = pivots.len();
        if let Some(k) = (rank..rows.len()).find(|&k| rows[k][col] != zero()) {
            rows.swap(rank, k);
            let pivot = rows[rank][col];
            for j in col..=s {
                rows[rank][j] = rows[rank][j] / pivot;
            }
            for i in 0..rows.len() {
                if i == rank {
                    continue;
                }
                let factor = rows[i][col];
                for j in col..=s {
                    rows[i][j] = rows[i][j] - factor * rows[rank][j];
                }
            }
            pivots.push(col);
        }
    }
    if rows
        .iter()
        .any(|r| r[..s].iter().all(|&x| x == zero()) && r[s] != zero())
    {
        return Err(Error::Inconsistent);
    }
    if pivots.len() < s {
        return Err(Error::Underdetermined);
    }
    let mut b = vec![zero(); s];
    for (i, &col) in pivots.iter().enumerate() {
        b[col] = rows[i][s];
    }
    Ok(b)
}
/// Coefficients of R(z)=1+z b^T (I-zA)^(-1) 1, ascending powers.
pub fn stability(a: &[Vec<Rational>], b: &[Rational]) -> Vec<Rational> {
    assert_eq!(a.len(), b.len());
    let mut v = vec![one(); b.len()];
    let mut r = vec![one()];
    for _ in 0..b.len() {
        r.push(dot(b, &v));
        v = a.iter().map(|row| dot(row, &v)).collect();
    }
    r
}
pub fn methods() -> Vec<(&'static str, usize, Vec<Vec<Rational>>)> {
    let z = zero();
    let o = one();
    vec![
        ("euler", 1, vec![vec![z]]),
        ("midpoint", 2, vec![vec![z, z], vec![q(1, 2), z]]),
        ("heun2", 2, vec![vec![z, z], vec![o, z]]),
        ("ralston2", 2, vec![vec![z, z], vec![q(2, 3), z]]),
        (
            "kutta3",
            3,
            vec![vec![z, z, z], vec![q(1, 2), z, z], vec![-o, q(2, 1), z]],
        ),
        (
            "rk4",
            4,
            vec![
                vec![z; 4],
                vec![q(1, 2), z, z, z],
                vec![z, q(1, 2), z, z],
                vec![z, z, o, z],
            ],
        ),
        (
            "rk38",
            4,
            vec![
                vec![z; 4],
                vec![q(1, 3), z, z, z],
                vec![q(-1, 3), o, z, z],
                vec![o, -o, o, z],
            ],
        ),
    ]
}
fn list(v: &[Rational]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
fn main() {
    for (name, p, a) in methods() {
        let b = derive_weights(&a, p).expect("valid method");
        let cs = conditions(&a, p + 1).unwrap();
        assert!(cs
            .iter()
            .filter(|c| c.order <= p)
            .all(|c| dot(&b, &c.weights) == c.target));
        assert!(cs
            .iter()
            .filter(|c| c.order == p + 1)
            .any(|c| dot(&b, &c.weights) != c.target));
        println!(
            "{name}\t{p}\t{}\t{}\t{}",
            list(&abscissae(&a)),
            list(&b),
            list(&stability(&a, &b))
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts() {
        let ts = trees(8);
        let counts: Vec<_> = (1..=8)
            .map(|p| ts.iter().filter(|t| t.order == p).count())
            .collect();
        assert_eq!(counts, [1, 1, 2, 4, 9, 20, 48, 115]);
    }
    #[test]
    fn known_weights_and_next_order_failure() {
        let expected = [
            vec![one()],
            vec![zero(), one()],
            vec![q(1, 2); 2],
            vec![q(1, 4), q(3, 4)],
            vec![q(1, 6), q(2, 3), q(1, 6)],
            vec![q(1, 6), q(1, 3), q(1, 3), q(1, 6)],
            vec![q(1, 8), q(3, 8), q(3, 8), q(1, 8)],
        ];
        for ((_, p, a), b) in methods().into_iter().zip(expected) {
            assert_eq!(derive_weights(&a, p), Ok(b.clone()));
            assert_eq!(derive_weights(&a, p + 1), Err(Error::Inconsistent));
            assert!(conditions(&a, p)
                .unwrap()
                .iter()
                .all(|c| dot(&b, &c.weights) == c.target));
        }
    }
    #[test]
    fn rejects_invalid_and_nonunique() {
        assert_eq!(derive_weights(&[], 1), Err(Error::InvalidMatrix));
        assert_eq!(derive_weights(&[vec![]], 1), Err(Error::InvalidMatrix));
        assert_eq!(derive_weights(&[vec![one()]], 1), Err(Error::InvalidMatrix));
        let a = &methods()[5].2;
        assert_eq!(derive_weights(a, 1), Err(Error::Underdetermined));
        assert_eq!(derive_weights(a, 0), Err(Error::UnsupportedOrder));
        assert_eq!(derive_weights(a, 9), Err(Error::UnsupportedOrder));
    }
    #[test]
    fn stability_alone_does_not_prove_nonlinear_order() {
        let a = vec![
            vec![zero(); 3],
            vec![one(), zero(), zero()],
            vec![zero(), one(), zero()],
        ];
        let b = vec![q(1, 2), q(1, 3), q(1, 6)];
        assert_eq!(stability(&a, &b), vec![one(), one(), q(1, 2), q(1, 6)]);
        assert!(conditions(&a, 3)
            .unwrap()
            .iter()
            .any(|c| dot(&b, &c.weights) != c.target));
        assert_eq!(derive_weights(&a, 3), Err(Error::Inconsistent));
    }
    #[test]
    fn rational_normalization() {
        assert_eq!(q(2, -4), q(-1, 2));
        assert_eq!(q(0, 9), zero());
        assert_eq!(q(1, 3) + q(1, 6), q(1, 2));
        assert_eq!(q(2, 3) / q(4, 9), q(3, 2));
    }
    #[test]
    #[should_panic]
    fn zero_denominator() {
        q(1, 0);
    }
    #[test]
    #[should_panic]
    fn overflow_never_wraps() {
        let _ = Rational::from(i128::MAX) + one();
    }
}
