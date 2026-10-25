//! The order conditions of HNW I §II.2, Theorem 2.13 / eq. (4.8), checked
//! numerically for a tableau: `Σ_j b_j Φ_j(t) = 1/γ(t)` for every rooted tree
//! `t` with at most `p` vertices.
//!
//! A tree is its multiset of subtrees attached to the root. For the stage
//! index `i`, the elementary weight is `Φ_i(•) = 1` for the single vertex
//! and `Φ_i(t) = Π_{subtrees u} Σ_j a_ij Φ_j(u)`; the density is
//! `γ(•) = 1`, `γ(t) = |t| Π_{subtrees u} γ(u)`.

/// A rooted tree as indices of its root's subtrees into the list of all trees.
struct Tree
{
  order: usize,
  subtrees: Vec<usize>,
}

/// Every rooted tree with at most `maximum_order` vertices, by order.
fn all_trees(maximum_order: usize) -> Vec<Tree>
{
  let mut trees = vec![Tree { order: 1, subtrees: Vec::new() }];
  for order in 2..=maximum_order
  {
    let existing = trees.len();
    let mut new_trees = Vec::new();
    let mut chosen = Vec::new();
    choose_subtrees(
      &trees, order - 1, existing, &mut chosen, &mut new_trees, order);
    trees.extend(new_trees);
  }
  trees
}

/// Appends every non-increasing sequence of tree indices `< bound` whose
/// orders sum to `remaining`; each sequence is one multiset of subtrees.
fn choose_subtrees(
  trees: &[Tree],
  remaining: usize,
  bound: usize,
  chosen: &mut Vec<usize>,
  output: &mut Vec<Tree>,
  order: usize)
{
  if remaining == 0
  {
    output.push(Tree { order, subtrees: chosen.clone() });
    return;
  }
  for index in (0..bound).rev()
  {
    if trees[index].order <= remaining
    {
      chosen.push(index);
      choose_subtrees(
        trees, remaining - trees[index].order, index + 1, chosen, output, order);
      chosen.pop();
    }
  }
}

/// The largest residual `|Σ_j b_j Φ_j(t) − 1/γ(t)|` over the trees of each
/// order `1..=maximum_order` (index 0 holds order 1).
pub fn residuals_by_order<const S: usize>(
  a: &[[f64; S]; S],
  b: &[f64; S],
  maximum_order: usize) -> Vec<f64>
{
  let trees = all_trees(maximum_order);
  let mut weights: Vec<[f64; S]> = Vec::with_capacity(trees.len());
  let mut densities: Vec<f64> = Vec::with_capacity(trees.len());
  let mut residuals = vec![0.0_f64; maximum_order];
  for tree in &trees
  {
    let mut phi = [1.0_f64; S];
    let mut gamma = tree.order as f64;
    for &subtree in &tree.subtrees
    {
      for i in 0..S
      {
        let mut inner = 0.0;
        for j in 0..S
        {
          inner += a[i][j] * weights[subtree][j];
        }
        phi[i] *= inner;
      }
      gamma *= densities[subtree];
    }
    let quadrature: f64 = (0..S).map(|j| b[j] * phi[j]).sum();
    let residual = (quadrature - 1.0 / gamma).abs();
    residuals[tree.order - 1] = residuals[tree.order - 1].max(residual);
    weights.push(phi);
    densities.push(gamma);
  }
  residuals
}

/// `c_i − Σ_j a_ij`, which must vanish for the stage abscissae to be the
/// times the stage values approximate.
pub fn row_sum_defect<const S: usize>(a: &[[f64; S]; S], c: &[f64; S]) -> f64
{
  (0..S)
    .map(|i| (c[i] - a[i].iter().sum::<f64>()).abs())
    .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests
{
  use super::*;

  /// The number of rooted trees with n vertices (OEIS A000081).
  #[test]
  fn counts_the_rooted_trees()
  {
    let trees = all_trees(8);
    let mut counts = [0usize; 8];
    for tree in &trees
    {
      counts[tree.order - 1] += 1;
    }
    assert_eq!(counts, [1, 1, 2, 4, 9, 20, 48, 115]);
    assert_eq!(trees.len(), 200);
  }

  /// Classical RK4 satisfies the 8 conditions of order ≤ 4 and fails order 5.
  #[test]
  fn classical_runge_kutta_has_order_four()
  {
    let a = [
      [0.0, 0.0, 0.0, 0.0],
      [0.5, 0.0, 0.0, 0.0],
      [0.0, 0.5, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
    ];
    let b = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
    let residuals = residuals_by_order(&a, &b, 5);
    assert!(residuals[..4].iter().all(|r| *r < 1.0e-15), "{residuals:?}");
    assert!(residuals[4] > 1.0e-3, "{residuals:?}");
  }
}
