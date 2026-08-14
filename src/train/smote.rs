//! SMOTE — Synthetic Minority Over-sampling Technique.
//!
//! For each minority-class sample, synthesize new points by linear
//! interpolation toward one of its k nearest neighbors (same class):
//!   x_new = x_i + λ·(x_nn − x_i),  λ ∈ [0,1]
//! Deterministic: a local xorshift PRNG (fixed seed) selects the source,
//! neighbor and λ, so the same input yields bit-identical synthetic samples.

/// SMOTE result.
pub struct SmoteResult {
    /// Synthetic feature vectors (minority class).
    pub synthetic_x: Vec<Vec<f64>>,
    /// Synthetic labels (all minority label).
    pub synthetic_y: Vec<f64>,
    /// Minority class label.
    pub minority_label: f64,
    /// Counts before/after.
    pub minority_before: usize,
    pub total_before: usize,
    pub total_after: usize,
}

fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rand_f64(st: &mut u64) -> f64 {
    (xorshift_next(st) as f64) / (u64::MAX as f64)
}

fn rand_usize(st: &mut u64, n: usize) -> usize {
    if n == 0 {
        0
    } else {
        (xorshift_next(st) as usize) % n
    }
}

/// Squared Euclidean distance.
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Run SMOTE on a binary-classification dataset.
///
/// - `x`: n_samples × n_features
/// - `y`: class labels (the least frequent class is oversampled)
/// - `k`: neighbors to consider (clamped to minority size − 1)
/// - `dup_ratio`: number of synthetic samples = floor(minority · dup_ratio)
pub fn smote(x: &[Vec<f64>], y: &[f64], k: usize, dup_ratio: f64) -> SmoteResult {
    assert_eq!(x.len(), y.len(), "x/y length mismatch");
    assert!(!x.is_empty(), "empty dataset");

    // minority class = least frequent
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();
    let counts: Vec<usize> = classes.iter().map(|&c| y.iter().filter(|&&v| v == c).count()).collect();
    let (mi, &minority_label) = classes
        .iter()
        .enumerate()
        .min_by_key(|(i, _)| counts[*i])
        .unwrap();
    let minority_count = counts[mi];

    let indices: Vec<usize> = (0..x.len()).filter(|&i| y[i] == minority_label).collect();
    let n_min = indices.len();
    let kk = k.min(n_min.saturating_sub(1)).max(1);

    // precompute per-source nearest neighbors (same class)
    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n_min);
    for (pos, &i) in indices.iter().enumerate() {
        let mut ds: Vec<(f64, usize)> = indices
            .iter()
            .enumerate()
            .filter(|(q, _)| *q != pos)
            .map(|(q, &j)| (dist2(&x[i], &x[j]), q))
            .collect();
        ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        neighbors.push(ds.into_iter().take(kk).map(|(_, q)| q).collect());
    }

    let n_synth = (minority_count as f64 * dup_ratio).floor() as usize;
    let mut synthetic_x = Vec::with_capacity(n_synth);
    let mut synthetic_y = Vec::with_capacity(n_synth);

    // deterministic local PRNG (fixed seed → reproducible output)
    let mut st = 0x5EED_CAFE_1234_5678u64;
    for _ in 0..n_synth {
        let src = rand_usize(&mut st, n_min);
        let nn = neighbors[src][rand_usize(&mut st, kk)];
        let lambda = rand_f64(&mut st);
        let (xi, xj) = (&x[indices[src]], &x[indices[nn]]);
        let mut row = Vec::with_capacity(xi.len());
        for d in 0..xi.len() {
            row.push(xi[d] + lambda * (xj[d] - xi[d]));
        }
        synthetic_x.push(row);
        synthetic_y.push(minority_label);
    }

    SmoteResult {
        synthetic_x,
        synthetic_y,
        minority_label,
        minority_before: minority_count,
        total_before: x.len(),
        total_after: x.len() + n_synth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imbalanced() -> (Vec<Vec<f64>>, Vec<f64>) {
        // 100 majority at (0,0) cluster, 10 minority at (5,5)
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..100 {
            x.push(vec![0.0, 0.0]);
            y.push(0.0);
        }
        for i in 0..10 {
            x.push(vec![5.0 + i as f64 * 0.1, 5.0 - i as f64 * 0.1]);
            y.push(1.0);
        }
        (x, y)
    }

    #[test]
    fn synthesizes_expected_count() {
        let (x, y) = imbalanced();
        let r = smote(&x, &y, 5, 1.0);
        assert_eq!(r.minority_before, 10);
        assert_eq!(r.synthetic_x.len(), 10); // dup_ratio 1.0 → 10 new
        assert_eq!(r.synthetic_y, vec![1.0; 10]);
        assert_eq!(r.total_before, 110);
        assert_eq!(r.total_after, 120);
    }

    #[test]
    fn synthetic_points_lie_near_minority_cluster() {
        let (x, y) = imbalanced();
        let r = smote(&x, &y, 5, 2.0);
        assert_eq!(r.synthetic_x.len(), 20);
        for row in &r.synthetic_x {
            // interpolation between minority points (near (5,5)) — far from majority (0,0)
            let d_maj = dist2(row, &[0.0, 0.0]);
            let d_min = dist2(row, &[5.0, 5.0]);
            assert!(d_min < d_maj, "synthetic point drifted: {row:?}");
        }
    }

    #[test]
    fn deterministic_output() {
        let (x, y) = imbalanced();
        let a = smote(&x, &y, 5, 3.0);
        let b = smote(&x, &y, 5, 3.0);
        assert_eq!(a.synthetic_x, b.synthetic_x);
        assert_eq!(a.synthetic_y, b.synthetic_y);
    }

    #[test]
    fn k_clamped_to_minority_size() {
        let (x, y) = imbalanced();
        let r = smote(&x, &y, 999, 1.0); // k >> minority size
        assert_eq!(r.synthetic_x.len(), 10);
    }

    #[test]
    fn dup_ratio_zero_no_synthesis() {
        let (x, y) = imbalanced();
        let r = smote(&x, &y, 5, 0.0);
        assert!(r.synthetic_x.is_empty());
        assert_eq!(r.total_after, r.total_before);
    }
}
