//! t-SNE (t-Distributed Stochastic Neighbor Embedding) — nonlinear
//! dimensionality reduction to 2D.
//!
//! Deterministic: Gaussian initialization via a local fixed-seed PRNG.
//! Prediction maps new points to the embedding of their nearest training
//! row (nearest-neighbor out-of-sample approximation — t-SNE has no closed
//! transform; this matches the community standard).

/// t-SNE result.
pub struct TsneResult {
    /// 2D embedding, n × 2.
    pub embedding: Vec<[f64; 2]>,
    /// Training rows (for nearest-neighbor mapping in predict).
    pub x: Vec<Vec<f64>>,
    /// Final KL divergence.
    pub kl: f64,
}

fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn randn(st: &mut u64) -> f64 {
    // Box-Muller from two uniforms
    let u1 = (xorshift_next(st) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    let u2 = (xorshift_next(st) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn embed_dist2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Binary-search per-point sigma for the target perplexity (Barnes-Hut-free).
/// `d` is the squared-distance row with `d[i] = inf` for the self entry —
/// that entry must be excluded from every sum (inf·0 = NaN).
fn binary_search_sigma(d: &[f64], perplexity: f64) -> f64 {
    let mut lo = 1e-10f64;
    let mut hi = 1e3f64;
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let mut z = 0.0f64;
        let mut dp = 0.0f64; // Σ d·p (finite entries only)
        for &x in d {
            if x.is_finite() {
                let e = (-x / (2.0 * mid * mid)).exp();
                z += e;
                dp += x * e;
            }
        }
        let z = z.max(1e-300);
        // H = ln Z + (1/2σ²)·Σ d·p (unnormalized p in the second term cancels
        // the Z; equivalently H = -Σ p ln p over normalized p)
        let h = z.ln() + dp / (2.0 * mid * mid) / z;
        let perp = h.exp();
        if perp > perplexity {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Run t-SNE.
pub fn train(
    x: &[Vec<f64>],
    perplexity: f64,
    max_iter: usize,
    lr: f64,
    momentum: f64,
) -> TsneResult {
    let n = x.len();
    assert!(n >= 2, "t-SNE needs >= 2 points");
    let perp = perplexity.clamp(2.0, (n as f64 - 1.0).max(2.0));

    // 1. symmetric high-dimensional affinities P
    let mut p = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut d: Vec<f64> = (0..n).map(|j| dist2(&x[i], &x[j])).collect();
        d[i] = f64::INFINITY; // exclude self
        let sigma = binary_search_sigma(&d, perp);
        for j in 0..n {
            if j != i {
                p[i][j] = (-d[j] / (2.0 * sigma * sigma)).exp();
            }
        }
        let s: f64 = p[i].iter().sum();
        for v in p[i].iter_mut() {
            *v /= s;
        }
    }
    for i in 0..n {
        for j in 0..n {
            p[i][j] = (p[i][j] + p[j][i]) / (2.0 * n as f64);
        }
    }

    // 2. deterministic Gaussian init
    let mut st = 0x5EED_CAFE_1234_5678u64;
    let mut y: Vec<[f64; 2]> = (0..n).map(|_| [randn(&mut st) * 1e-4, randn(&mut st) * 1e-4]).collect();
    let mut vel = vec![[0.0f64; 2]; n];

    // early exaggeration (sklearn: p *= 12 for the first 250 iterations)
    let exaggerate_iters = 250usize.min(max_iter);
    let mut kl = f64::INFINITY;
    for it in 0..max_iter {
        let exagg = if it < exaggerate_iters { 12.0 } else { 1.0 };
        // 3. low-dimensional affinities Q (t-distribution, no normalization
        // of the constant 1/(1+d) needed beyond Q sum)
        let mut q = vec![vec![0.0f64; n]; n];
        let mut q_sum = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d = embed_dist2(&y[i], &y[j]);
                    q[i][j] = 1.0 / (1.0 + d);
                    q_sum += q[i][j];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                q[i][j] /= q_sum;
            }
        }

        // 4. KL gradient: 4·Σ_j (p·exagg − q)·(y_i−y_j)·(1+||y_i−y_j||²)⁻¹
        kl = 0.0;
        let mut grad = vec![[0.0f64; 2]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = embed_dist2(&y[i], &y[j]);
                let inv = 1.0 / (1.0 + d);
                let diff = p[i][j] * exagg - q[i][j];
                let c = 4.0 * diff * inv;
                grad[i][0] += c * (y[i][0] - y[j][0]);
                grad[i][1] += c * (y[i][1] - y[j][1]);
                let pij = p[i][j] * exagg;
                if pij > 0.0 {
                    kl += pij * (pij / q[i][j].max(1e-300)).ln();
                }
            }
        }

        // 5. gradient descent with momentum (sklearn: 0.5 during early
        // exaggeration, 0.8 afterwards)
        let mom = if it < exaggerate_iters { 0.5 } else { momentum };
        let scale = (n as f64 - 1.0) / n as f64;
        for i in 0..n {
            vel[i][0] = mom * vel[i][0] - lr * grad[i][0] * scale;
            vel[i][1] = mom * vel[i][1] - lr * grad[i][1] * scale;
            y[i][0] += vel[i][0];
            y[i][1] += vel[i][1];
        }
    }

    TsneResult {
        embedding: y,
        x: x.to_vec(),
        kl,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_blobs_2d() -> Vec<Vec<f64>> {
        // 3 separated 2D blobs (60 pts) — t-SNE should keep them apart
        let mut st = 42u64;
        let mut x = Vec::new();
        let centers = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        for i in 0..60 {
            let c = centers[i % 3];
            let mut row = vec![c[0] + randn(&mut st) * 0.5, c[1] + randn(&mut st) * 0.5];
            // normalize a bit so perplexity search is well-conditioned
            row[0] *= 0.1;
            row[1] *= 0.1;
            x.push(row);
        }
        x
    }

    #[test]
    fn embedding_keeps_blobs_separated() {
        let x = three_blobs_2d();
        let r = train(&x, 15.0, 1000, 200.0, 0.8);
        assert_eq!(r.embedding.len(), 60);
        // within-blob pair distances must be smaller than between-blob
        // distances on average
        let mut within = 0.0f64;
        let mut between = 0.0f64;
        let mut wc = 0;
        let mut bc = 0;
        for i in 0..60 {
            for j in (i + 1)..60 {
                let d = embed_dist2(&r.embedding[i], &r.embedding[j]);
                if i % 3 == j % 3 {
                    within += d;
                    wc += 1;
                } else {
                    between += d;
                    bc += 1;
                }
            }
        }
        let w = within / wc as f64;
        let b = between / bc as f64;
        assert!(w < b * 0.5, "blobs not separated: within={w} between={b} kl={}", r.kl);
        assert!(r.kl.is_finite());
    }

    #[test]
    fn sigma_search_sane() {
        // 60 points in 3 blobs, spacing 1.0 → sigma must be ~0.1..0.5, not 1e3
        let x = three_blobs_2d();
        let d: Vec<f64> = (0..x.len()).map(|j| dist2(&x[0], &x[j])).collect();
        let s = binary_search_sigma(&d, 15.0);
        assert!(s > 1e-4 && s < 100.0, "sigma out of range: {s}");
    }

    #[test]
    fn deterministic() {
        let x = three_blobs_2d();
        let a = train(&x, 15.0, 50, 200.0, 0.8);
        let b = train(&x, 15.0, 50, 200.0, 0.8);
        assert_eq!(a.embedding, b.embedding);
    }
}
