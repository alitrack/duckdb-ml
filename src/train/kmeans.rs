//! K-Means clustering — Lloyd's algorithm with k-means++ initialization
//!
//! Pure Rust, zero external dependencies beyond std.

/// K-Means clustering result
pub struct KMeansResult {
    /// Cluster centroids: k rows × n_features columns
    pub centroids: Vec<Vec<f64>>,
    /// Which cluster each training sample belongs to
    pub labels: Vec<usize>,
    /// Number of iterations until convergence
    pub iterations: usize,
}

/// Run K-Means on data
///
/// - `x`: n_samples × n_features
/// - `k`: number of clusters
/// - `max_iters`: maximum Lloyd iterations
/// - `tol`: stop when centroid shift is below this threshold
/// - `seed`: optional PRNG seed (0 = use xorshift state)
pub fn train(x: &[Vec<f64>], k: usize, max_iters: usize, tol: f64) -> KMeansResult {
    let n_samples = x.len();
    let n_features = if n_samples > 0 { x[0].len() } else { 0 };

    assert!(k > 0 && k <= n_samples, "k must be in 1..=n_samples");
    assert!(n_features > 0, "empty dataset");

    // 1. Initialize centroids with k-means++
    let mut centroids = kmeans_plusplus(x, k);

    let mut labels = vec![0usize; n_samples];
    let mut iterations = 0;

    // 2. Lloyd iteration
    for _ in 0..max_iters {
        iterations += 1;

        // Assign each point to nearest centroid
        let mut changed = false;
        for (i, row) in x.iter().enumerate() {
            let new_label = nearest_centroid(row, &centroids);
            if new_label != labels[i] {
                labels[i] = new_label;
                changed = true;
            }
        }

        // If no label changed, converged
        if !changed {
            break;
        }

        // Update centroids: mean of assigned points
        let old_centroids = centroids.clone();
        let mut counts = vec![0usize; k];
        centroids = vec![vec![0.0f64; n_features]; k];

        for (i, row) in x.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..n_features {
                centroids[c][j] += row[j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for val in centroids[c].iter_mut() {
                    *val *= inv;
                }
            } else {
                // Empty cluster: reinitialize to a random point far from others
                let idx = random_point_far_from_centroids(x, &old_centroids);
                centroids[c] = x[idx].clone();
            }
        }

        // Check convergence by centroid movement
        let max_shift: f64 = centroids
            .iter()
            .zip(old_centroids.iter())
            .map(|(new, old)| {
                new.iter()
                    .zip(old.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0f64, f64::max);

        if max_shift < tol {
            break;
        }
    }

    KMeansResult {
        centroids,
        labels,
        iterations,
    }
}

/// Find the index of the nearest centroid to a point (Euclidean distance)
#[inline]
pub fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut best_dist = f64::MAX;

    for (i, c) in centroids.iter().enumerate() {
        let dist: f64 = point
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    best
}

/// k-means++ centroid initialization
fn kmeans_plusplus(x: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut centroids = Vec::with_capacity(k);

    // First centroid: uniform random
    centroids.push(x[rand_usize(n)].clone());

    // Remaining centroids: weighted by squared distance to nearest existing centroid
    let mut dists = vec![f64::MAX; n];

    for _ in 1..k {
        // Update distances to nearest existing centroid
        for (i, row) in x.iter().enumerate() {
            let last = centroids.last().unwrap();
            let d: f64 = row
                .iter()
                .zip(last.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if d < dists[i] {
                dists[i] = d;
            }
        }

        // Weighted random selection: probability ∝ distance²
        let total: f64 = dists.iter().sum();
        if total < 1e-12 {
            // All points coincide with centroids — pick random
            centroids.push(x[rand_usize(n)].clone());
            continue;
        }

        let mut r = rand_f64() * total;
        let mut idx = 0;
        for (i, &d) in dists.iter().enumerate() {
            r -= d;
            if r <= 0.0 {
                idx = i;
                break;
            }
        }
        centroids.push(x[idx].clone());
    }

    centroids
}

/// Pick a random point that is far from existing centroids (for empty cluster reinit)
fn random_point_far_from_centroids(x: &[Vec<f64>], centroids: &[Vec<f64>]) -> usize {
    let n = x.len();
    let mut max_dist = 0.0f64;
    let mut best = 0usize;

    // Sample a few random points and pick the farthest from any centroid
    for _ in 0..10.min(n) {
        let idx = rand_usize(n);
        let min_d: f64 = centroids
            .iter()
            .map(|c| {
                x[idx]
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
            })
            .fold(f64::MAX, f64::min);
        if min_d > max_dist {
            max_dist = min_d;
            best = idx;
        }
    }
    best
}

// ——— PRNG ———

static XORSHIFT_STATE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x9E37_79B9_7F4A_7C15);

#[inline]
fn xorshift_u64() -> u64 {
    use std::sync::atomic::Ordering;
    let mut x = XORSHIFT_STATE.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    XORSHIFT_STATE.store(x, Ordering::Relaxed);
    x
}

#[inline]
fn rand_f64() -> f64 {
    let v = xorshift_u64();
    (v as f64) / (u64::MAX as f64)
}

#[inline]
fn rand_usize(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    (xorshift_u64() as usize) % n
}

// ——— Serialization for model storage ———

/// Serialize centroids to bytes: k (u32) + n_features (u32) + centroids (f64[k*n_features])
pub fn serialize_centroids(centroids: &[Vec<f64>]) -> Vec<u8> {
    let k = centroids.len();
    let nf = if k > 0 { centroids[0].len() } else { 0 };
    let mut buf = Vec::with_capacity(8 + k * nf * 8);
    buf.extend_from_slice(&(k as u32).to_le_bytes());
    buf.extend_from_slice(&(nf as u32).to_le_bytes());
    for row in centroids {
        for &v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Deserialize centroids from bytes
pub fn deserialize_centroids(data: &[u8]) -> Option<(Vec<Vec<f64>>, usize, usize)> {
    if data.len() < 8 {
        return None;
    }
    let k = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let nf = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
    let expected = 8 + k * nf * 8;
    if data.len() < expected {
        return None;
    }
    let mut centroids = Vec::with_capacity(k);
    let mut pos = 8;
    for _ in 0..k {
        let mut row = Vec::with_capacity(nf);
        for _ in 0..nf {
            let bytes: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
            row.push(f64::from_le_bytes(bytes));
            pos += 8;
        }
        centroids.push(row);
    }
    Some((centroids, k, nf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_serialization_roundtrip() {
        let centroids = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let bytes = serialize_centroids(&centroids);
        let (recovered, k, nf) = deserialize_centroids(&bytes).unwrap();
        assert_eq!(k, 2);
        assert_eq!(nf, 3);
        assert_eq!(recovered, centroids);
    }

    #[test]
    fn test_kmeans_two_clusters() {
        // Two well-separated blobs
        let mut x = Vec::new();
        for _ in 0..50 {
            x.push(vec![1.0 + rand_f64() * 0.5, 1.0 + rand_f64() * 0.5]);
            x.push(vec![5.0 + rand_f64() * 0.5, 5.0 + rand_f64() * 0.5]);
        }
        let result = train(&x, 2, 50, 1e-6);

        assert_eq!(result.centroids.len(), 2);
        assert!(result.iterations <= 50);

        // Centroids should be near (1.25, 1.25) and (5.25, 5.25)
        let c0_near_1 = result.centroids[0][0] < 3.0;
        let c1_near_1 = result.centroids[1][0] < 3.0;
        assert!(c0_near_1 != c1_near_1, "centroids not separated");
    }

    #[test]
    fn test_kmeans_three_1d_clusters() {
        // Three clusters along a line
        let mut x = Vec::new();
        for _ in 0..30 {
            x.push(vec![0.0]);
            x.push(vec![10.0]);
            x.push(vec![20.0]);
        }
        let result = train(&x, 3, 50, 1e-4);

        assert_eq!(result.centroids.len(), 3);
        let mut cs: Vec<f64> = result.centroids.iter().map(|c| c[0]).collect();
        cs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((cs[0] - 0.0).abs() < 0.01, "c0={}", cs[0]);
        assert!((cs[1] - 10.0).abs() < 0.01, "c1={}", cs[1]);
        assert!((cs[2] - 20.0).abs() < 0.01, "c2={}", cs[2]);
    }
}
