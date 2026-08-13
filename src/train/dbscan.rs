//! DBSCAN clustering via linfa-clustering (0.8.1, MIT).
//!
//! Density-based clustering: points are grouped into arbitrary-shape clusters
//! by an `eps`-neighborhood reachability rule; points without enough
//! `min_points` neighbors are marked as noise. Matches MADlib's `dbscan` /
//! scikit-learn semantics.

use linfa::traits::Transformer;
use linfa_clustering::Dbscan;
use ndarray::Array2;

/// One cluster: its points' mean (representative vector) and size.
#[derive(Debug, Clone, PartialEq)]
pub struct DbscanCluster {
    pub label: usize,
    pub count: usize,
    pub mean: Vec<f64>,
}

/// Result of a DBSCAN fit.
#[derive(Debug, Clone, PartialEq)]
pub struct DbscanTrainResult {
    pub clusters: Vec<DbscanCluster>,
    pub noise_count: usize,
}

/// Fit DBSCAN over `x` (n_samples × n_features).
///
/// `eps`: neighborhood radius (> 0); `min_points`: core-point density (>= 1).
/// Points closer than `eps` to a core point join its cluster; everything else
/// is noise (noise points are excluded from every cluster's mean).
pub fn train(x: &[Vec<f64>], eps: f64, min_points: usize) -> Result<DbscanTrainResult, String> {
    let n = x.len();
    let n_features = if n > 0 { x[0].len() } else { 0 };
    if n == 0 {
        return Err("dbscan: empty dataset".into());
    }
    if n_features == 0 {
        return Err("dbscan: features must have at least one column".into());
    }
    if !(eps > 0.0 && eps.is_finite()) {
        return Err("dbscan: eps must be a positive finite number".into());
    }
    if min_points < 1 {
        return Err("dbscan: min_points must be >= 1".into());
    }

    let flat: Vec<f64> = x.iter().flatten().copied().collect();
    let arr = Array2::from_shape_vec((n, n_features), flat)
        .map_err(|e| format!("dbscan: array build failed: {e}"))?;

    let labels: ndarray::Array1<Option<usize>> = Dbscan::params(min_points)
        .tolerance(eps)
        .transform(&arr)
        .map_err(|e| format!("dbscan: {e}"))?;

    // Aggregate per-cluster sums/counts; count noise.
    let mut sums: Vec<Vec<f64>> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut noise = 0usize;
    for (i, label) in labels.indexed_iter() {
        if let Some(cid) = label {
            let cid = *cid;
            if cid >= sums.len() {
                sums.resize(cid + 1, vec![0.0; n_features]);
                counts.resize(cid + 1, 0);
            }
            for (s, v) in sums[cid].iter_mut().zip(x[i].iter()) {
                *s += *v;
            }
            counts[cid] += 1;
        } else {
            noise += 1;
        }
    }

    let clusters = sums
        .into_iter()
        .zip(counts)
        .enumerate()
        .map(|(label, (sum, count))| DbscanCluster {
            label,
            count,
            mean: sum.iter().map(|s| s / count as f64).collect(),
        })
        .collect();

    Ok(DbscanTrainResult {
        clusters,
        noise_count: noise,
    })
}

/// Serialize to bytes: n_clusters (u32) + n_features (u32) + eps (f64)
/// + per cluster [count (u64) + mean (f64 × n_features)].
pub fn serialize(clusters: &[DbscanCluster], n_features: usize, eps: f64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(clusters.len() as u32).to_le_bytes());
    out.extend_from_slice(&(n_features as u32).to_le_bytes());
    out.extend_from_slice(&eps.to_le_bytes());
    for c in clusters {
        out.extend_from_slice(&(c.count as u64).to_le_bytes());
        for v in &c.mean {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Deserialize a DBSCAN model blob produced by [`serialize`].
/// Returns `(clusters, n_features, eps)`.
pub fn deserialize(blob: &[u8]) -> Option<(Vec<DbscanCluster>, usize, f64)> {
    let mut pos = 0usize;
    let read_u32 = |blob: &[u8], pos: &mut usize| -> Option<u32> {
        if *pos + 4 > blob.len() {
            return None;
        }
        let v = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?);
        *pos += 4;
        Some(v)
    };
    let read_u64 = |blob: &[u8], pos: &mut usize| -> Option<u64> {
        if *pos + 8 > blob.len() {
            return None;
        }
        let v = u64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
        *pos += 8;
        Some(v)
    };
    let read_f64 = |blob: &[u8], pos: &mut usize| -> Option<f64> {
        if *pos + 8 > blob.len() {
            return None;
        }
        let v = f64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
        *pos += 8;
        Some(v)
    };

    let n_clusters = read_u32(blob, &mut pos)? as usize;
    let n_features = read_u32(blob, &mut pos)? as usize;
    let eps = read_f64(blob, &mut pos)?;
    let mut clusters = Vec::with_capacity(n_clusters);
    for label in 0..n_clusters {
        let count = read_u64(blob, &mut pos)? as usize;
        let mut mean = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            mean.push(read_f64(blob, &mut pos)?);
        }
        clusters.push(DbscanCluster { label, count, mean });
    }
    Some((clusters, n_features, eps))
}

/// Nearest-cluster label for a feature vector (MADlib-style simplified
/// prediction: assign to the closest cluster representative).
pub fn nearest_cluster(features: &[f64], clusters: &[DbscanCluster]) -> f64 {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, c) in clusters.iter().enumerate() {
        let d = squared_distance(features, &c.mean);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best as f64
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_blobs_with_noise() {
        // Two dense blobs + one far-away outlier; eps small enough to keep
        // blobs separate.
        let mut x: Vec<Vec<f64>> = Vec::new();
        for i in 0..20 {
            x.push(vec![i as f64 * 0.1, 0.0]); // blob A along x
            x.push(vec![5.0 + i as f64 * 0.1, 5.0]); // blob B
        }
        x.push(vec![100.0, 100.0]); // noise
        let r = train(&x, 0.35, 3).unwrap();
        assert_eq!(r.clusters.len(), 2, "clusters: {:?}", r.clusters);
        assert_eq!(r.noise_count, 1);
        // blob A mean ≈ (0.95, 0), blob B mean ≈ (5.95, 5)
        for c in &r.clusters {
            assert_eq!(c.count, 20);
            if c.mean[1] < 1.0 {
                assert!((c.mean[0] - 0.95).abs() < 1e-6);
            } else {
                assert!((c.mean[0] - 5.95).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn all_noise_when_isolated() {
        let x = vec![vec![0.0], vec![10.0], vec![20.0]];
        let r = train(&x, 0.5, 2).unwrap();
        assert!(r.clusters.is_empty());
        assert_eq!(r.noise_count, 3);
    }

    #[test]
    fn validates_params() {
        assert!(train(&[vec![1.0]], 0.0, 2).is_err());
        assert!(train(&[vec![1.0]], 0.5, 0).is_err());
        assert!(train(&[], 0.5, 2).is_err());
    }

    #[test]
    fn serialize_roundtrip() {
        let x = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.2, 0.0]];
        let r = train(&x, 0.3, 2).unwrap();
        let blob = serialize(&r.clusters, 2, 0.3);
        let (clusters, nf, eps) = deserialize(&blob).unwrap();
        assert_eq!(nf, 2);
        assert_eq!(eps, 0.3);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].count, 3);
        assert!((clusters[0].mean[0] - 0.1).abs() < 1e-9);
        assert_eq!(clusters[0].mean[1], 0.0);
        assert!(deserialize(&blob[..4]).is_none());
    }

    #[test]
    fn nearest_cluster_assigns_closest() {
        let clusters = vec![
            DbscanCluster {
                label: 0,
                count: 2,
                mean: vec![0.0, 0.0],
            },
            DbscanCluster {
                label: 1,
                count: 2,
                mean: vec![10.0, 10.0],
            },
        ];
        assert_eq!(nearest_cluster(&[0.1, -0.1], &clusters), 0.0);
        assert_eq!(nearest_cluster(&[9.9, 10.1], &clusters), 1.0);
    }
}
