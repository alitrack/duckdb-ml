//! Fuzzy c-Means (FCM) clustering — soft-assignment generalization of k-means.
//!
//! Minimizes Σᵢ Σⱼ uᵢⱼᵐ ‖xᵢ − cⱼ‖² with uᵢⱼ ∈ [0,1], Σⱼ uᵢⱼ = 1, m = fuzziness
//! exponent (m → 1 recovers hard k-means). Standard alternating updates:
//!   uᵢⱼ = (Σₖ (dᵢⱼ/dᵢₖ)^(2/(m−1)))⁻¹      (dᵢⱼ = squared distance xᵢ→cⱼ)
//!   cⱼ  = Σᵢ uᵢⱼᵐ xᵢ / Σᵢ uᵢⱼᵐ
//! Initialization reuses k-means++ (deterministic xorshift PRNG).

use super::kmeans::kmeans_plusplus;

/// Fuzzy c-Means result
pub struct FcmResult {
    /// Cluster centroids: k rows × n_features columns
    pub centroids: Vec<Vec<f64>>,
    /// Membership matrix (n_samples × k), rows sum to 1
    pub memberships: Vec<Vec<f64>>,
    /// Number of iterations until convergence
    pub iterations: usize,
}

/// Run fuzzy c-Means.
///
/// - `x`: n_samples × n_features
/// - `k`: number of clusters
/// - `m`: fuzziness exponent (> 1; 2 is the classic default)
/// - `max_iters`: maximum update rounds
/// - `tol`: stop when max |Δu| < tol
pub fn train(x: &[Vec<f64>], k: usize, m: f64, max_iters: usize, tol: f64) -> FcmResult {
    let n_samples = x.len();
    let n_features = if n_samples > 0 { x[0].len() } else { 0 };
    assert!(k > 0 && k <= n_samples, "k must be in 1..=n_samples");
    assert!(n_features > 0, "empty dataset");
    let m = if m > 1.0 { m } else { 2.0 };

    // 1. k-means++ initialization
    let mut centroids = kmeans_plusplus(x, k);

    // 2. Alternating membership/centroid updates
    let mut memberships = vec![vec![1.0 / k as f64; k]; n_samples];
    let exp = 2.0 / (m - 1.0); // exponent in the membership formula

    let mut iterations = 0;
    for _ in 0..max_iters {
        iterations += 1;

        // a. distances + membership update
        let mut new_u = vec![vec![0.0f64; k]; n_samples];
        for (i, row) in x.iter().enumerate() {
            let mut d = vec![0.0f64; k];
            let mut any_zero = false;
            for (j, c) in centroids.iter().enumerate() {
                let dist: f64 = row
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                d[j] = dist;
                if dist < 1e-15 {
                    any_zero = true;
                }
            }
            if any_zero {
                // exact hit: full membership to the nearest zero-distance cluster
                let mut best = 0usize;
                let mut best_d = f64::MAX;
                for (j, &dj) in d.iter().enumerate() {
                    if dj < best_d {
                        best_d = dj;
                        best = j;
                    }
                }
                new_u[i][best] = 1.0;
            } else {
                let mut inv_sum = 0.0f64;
                for j in 0..k {
                    let mut s = 0.0f64;
                    for dj in &d {
                        s += (d[j] / dj).powf(exp);
                    }
                    new_u[i][j] = if s > 0.0 { 1.0 / s } else { 0.0 };
                    inv_sum += new_u[i][j];
                }
                // normalize (numerical guard)
                if inv_sum > 0.0 {
                    for u in new_u[i].iter_mut() {
                        *u /= inv_sum;
                    }
                }
            }
        }

        // b. centroid update: weighted mean with uᵐ
        let mut new_centroids = vec![vec![0.0f64; n_features]; k];
        let mut weight_sum = vec![0.0f64; k];
        for (i, row) in x.iter().enumerate() {
            for j in 0..k {
                let w = new_u[i][j].powf(m);
                weight_sum[j] += w;
                for f in 0..n_features {
                    new_centroids[j][f] += w * row[f];
                }
            }
        }
        for j in 0..k {
            if weight_sum[j] > 0.0 {
                let inv = 1.0 / weight_sum[j];
                for v in new_centroids[j].iter_mut() {
                    *v *= inv;
                }
            } else {
                new_centroids[j] = centroids[j].clone(); // empty: keep previous
            }
        }

        // c. convergence: max |Δu|
        let max_delta: f64 = memberships
            .iter()
            .zip(new_u.iter())
            .map(|(old_row, new_row)| {
                old_row
                    .iter()
                    .zip(new_row.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max)
            })
            .fold(0.0f64, f64::max);

        memberships = new_u;
        centroids = new_centroids;

        if max_delta < tol {
            break;
        }
    }

    FcmResult {
        centroids,
        memberships,
        iterations,
    }
}

/// Nearest centroid index (hard label) — same convention as k-means.
pub fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> usize {
    super::kmeans::nearest_centroid(point, centroids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_gaussians() -> Vec<Vec<f64>> {
        let mut x = Vec::new();
        for _ in 0..3 {
            for i in 0..40 {
                for j in 0..40 {
                    // deterministic 2D grid clouds centered (0,0) and (6,0)
                    x.push(vec![(i as f64 - 20.0) * 0.2, (j as f64 - 20.0) * 0.2]);
                }
            }
        }
        for _ in 0..3 {
            for i in 0..40 {
                for j in 0..40 {
                    x.push(vec![(i as f64 - 20.0) * 0.2 + 6.0, (j as f64 - 20.0) * 0.2]);
                }
            }
        }
        x
    }

    #[test]
    fn separates_two_clouds() {
        let x = two_gaussians();
        let r = train(&x, 2, 2.0, 200, 1e-5);
        // centroids near (0,0) and (6,0)
        let c0 = &r.centroids[0];
        let c1 = &r.centroids[1];
        let dx = (c0[0] - c1[0]).abs();
        assert!(dx > 4.0, "centroids too close: {c0:?} {c1:?}");
        assert!(r.memberships.len() == x.len());
        // memberships sum to 1 per row
        for row in &r.memberships {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "row sum {s}");
        }
    }

    #[test]
    fn converges_deterministically() {
        let x = two_gaussians();
        let a = train(&x, 3, 2.0, 200, 1e-6);
        let b = train(&x, 3, 2.0, 200, 1e-6);
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.memberships, b.memberships);
    }

    #[test]
    fn hard_label_matches_nearest_centroid() {
        let x = two_gaussians();
        let r = train(&x, 2, 2.0, 100, 1e-5);
        // argmax membership should equal nearest-centroid label
        for (i, row) in r.memberships.iter().enumerate() {
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(argmax, nearest_centroid(&x[i], &r.centroids));
        }
    }
}
