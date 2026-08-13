//! Multinomial logistic regression (softmax) — MADlib `multilogistic` counterpart.
//!
//! Full-batch gradient descent over the softmax cross-entropy loss. Classes
//! are any distinct numeric labels (e.g. 5, 7, 9); they are mapped to 0..k-1
//! internally and `predict` returns the original label values.
//!
//! Blob layout (little-endian, hand-rolled like kmeans/dbscan):
//!   u32 k · u32 d · k × (d+1) f64 weight rows (intercept last)

/// Softmax multinomial logistic regression weights.
pub struct MultinomialModel {
    /// Unique class labels, sorted ascending; index i ↔ row i of `weights`.
    pub classes: Vec<f64>,
    /// k × (d+1) weight matrix; row c = [w_c0..w_c(d-1), intercept_c].
    pub weights: Vec<Vec<f64>>,
    pub n_features: usize,
}

/// Train a softmax classifier. `y` must contain finite numeric class labels.
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_epochs: usize,
) -> Result<MultinomialModel, String> {
    if x.is_empty() || x.len() != y.len() {
        return Err("empty or mismatched data".into());
    }
    let n = x.len();
    let d = x[0].len();
    if d == 0 {
        return Err("zero features".into());
    }
    if lr <= 0.0 {
        return Err("lr must be > 0".into());
    }

    // unique sorted classes → 0..k-1 index (f64 has no Ord; sort via partial_cmp)
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();
    if classes.len() < 2 {
        return Err(format!("need ≥ 2 classes, got {}", classes.len()));
    }
    let k = classes.len();
    let y_idx: Vec<usize> = y
        .iter()
        .map(|v| {
            classes
                .iter()
                .position(|c| (c - v).abs() < 1e-12)
                .expect("class in set")
        })
        .collect();

    let d1 = d + 1;
    // initialize all weights to 0 (with small random jitter not needed; 0 init
    // is fine for softmax and keeps training deterministic)
    let mut weights = vec![vec![0.0f64; d1]; k];

    let mut prev_loss = f64::MAX;
    for _epoch in 0..max_epochs {
        // per-class score sums for this batch (full batch GD)
        let mut grads = vec![vec![0.0f64; d1]; k];
        let mut total_loss = 0.0f64;

        for i in 0..n {
            // z_c = w_c · x_i (intercept last)
            let mut z = vec![0.0f64; k];
            for c in 0..k {
                let mut s = weights[c][d];
                for j in 0..d {
                    s += weights[c][j] * x[i][j];
                }
                z[c] = s;
            }
            // numerically stable softmax
            let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut exps = vec![0.0f64; k];
            let mut sum = 0.0f64;
            for c in 0..k {
                exps[c] = (z[c] - m).exp();
                sum += exps[c];
            }
            let p: Vec<f64> = exps.iter().map(|e| e / sum).collect();

            let yi = y_idx[i];
            total_loss += -p[yi].ln().max(-1e6); // clamp: p ≥ 1e-300 guaranteed by softmax

            // gradient: (p_c - I(y=c)) * x
            for c in 0..k {
                let err = p[c] - if c == yi { 1.0 } else { 0.0 };
                for j in 0..d {
                    grads[c][j] += err * x[i][j];
                }
                grads[c][d] += err;
            }
        }

        total_loss /= n as f64;
        for c in 0..k {
            for j in 0..d1 {
                weights[c][j] -= lr * grads[c][j] / n as f64;
            }
        }

        if (prev_loss - total_loss).abs() < 1e-7 {
            break;
        }
        prev_loss = total_loss;
    }

    Ok(MultinomialModel {
        classes,
        weights,
        n_features: d,
    })
}

/// Predict the class label (original value) for one feature row.
pub fn predict_one(m: &MultinomialModel, features: &[f64]) -> f64 {
    let d = m.n_features;
    let mut best_c = 0usize;
    let mut best_z = f64::NEG_INFINITY;
    for c in 0..m.classes.len() {
        let mut s = m.weights[c][d];
        for (j, f) in features.iter().enumerate() {
            s += m.weights[c][j] * f;
        }
        if s > best_z {
            best_z = s;
            best_c = c;
        }
    }
    m.classes[best_c]
}

// ── serialization ──

pub fn serialize(m: &MultinomialModel) -> Vec<u8> {
    let k = m.classes.len();
    let d = m.n_features;
    let mut out = Vec::with_capacity(8 + k * (d + 1) * 8);
    out.extend_from_slice(&(k as u32).to_le_bytes());
    out.extend_from_slice(&(d as u32).to_le_bytes());
    for c in 0..k {
        out.extend_from_slice(&m.classes[c].to_le_bytes());
        for j in 0..=d {
            out.extend_from_slice(&m.weights[c][j].to_le_bytes());
        }
    }
    out
}

pub fn deserialize(blob: &[u8]) -> Result<MultinomialModel, String> {
    let need = |i: usize, n: usize| -> Result<(), String> {
        if blob.len() < i + n {
            return Err("truncated blob".into());
        }
        Ok(())
    };
    need(0, 8)?;
    let k = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(blob[4..8].try_into().unwrap()) as usize;
    if k == 0 || d == 0 || k > 100_000 {
        return Err("implausible k/d".into());
    }
    let row_bytes = (1 + d + 1) * 8; // class label + d weights + intercept
    need(8, k * row_bytes)?;
    let mut classes = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);
    for c in 0..k {
        let base = 8 + c * row_bytes;
        let mut label = [0u8; 8];
        label.copy_from_slice(&blob[base..base + 8]);
        classes.push(f64::from_le_bytes(label));
        let mut row = Vec::with_capacity(d + 1);
        for j in 0..=d {
            let mut b = [0u8; 8];
            let off = base + 8 + j * 8;
            b.copy_from_slice(&blob[off..off + 8]);
            row.push(f64::from_le_bytes(b));
        }
        weights.push(row);
    }
    Ok(MultinomialModel {
        classes,
        weights,
        n_features: d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_class_separable() {
        // 3 blobs on the x-axis: ~-5 → 0, ~0 → 1, ~5 → 2
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..20 {
            let t = i as f64;
            x.push(vec![-5.0 + t * 0.05, 1.0]);
            y.push(0.0);
            x.push(vec![t * 0.05, 1.0]);
            y.push(1.0);
            x.push(vec![5.0 + t * 0.05, 1.0]);
            y.push(2.0);
        }
        let m = train(&x, &y, 0.1, 1000).unwrap();
        assert_eq!(m.classes, vec![0.0, 1.0, 2.0]);
        // exact classification of the three centroids
        assert_eq!(predict_one(&m, &[-5.0, 1.0]), 0.0);
        assert_eq!(predict_one(&m, &[0.0, 1.0]), 1.0);
        assert_eq!(predict_one(&m, &[5.0, 1.0]), 2.0);
        // non-consecutive labels roundtrip
        let m2 = deserialize(&serialize(&m)).unwrap();
        assert_eq!(m2.classes, vec![0.0, 1.0, 2.0]);
        assert_eq!(predict_one(&m2, &[-5.0, 1.0]), 0.0);
        assert_eq!(predict_one(&m2, &[5.0, 1.0]), 2.0);
    }

    #[test]
    fn non_consecutive_labels() {
        // labels 10 / 20 / 30
        let x = vec![
            vec![-3.0],
            vec![-2.5],
            vec![2.5],
            vec![3.0],
            vec![-1.0],
            vec![1.0],
        ];
        let y = vec![10.0, 10.0, 30.0, 30.0, 10.0, 30.0];
        let m = train(&x, &y, 0.2, 1500).unwrap();
        assert_eq!(m.classes, vec![10.0, 30.0]);
        assert_eq!(predict_one(&m, &[-3.0]), 10.0);
        assert_eq!(predict_one(&m, &[3.0]), 30.0);
        assert_eq!(predict_one(&m, &[-2.0]), 10.0);
        assert_eq!(predict_one(&m, &[2.0]), 30.0);
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[], &[], 0.1, 100).is_err());
        assert!(train(&[vec![1.0]], &[0.0, 1.0], 0.1, 100).is_err());
        assert!(train(&[vec![1.0]], &[0.0], 0.1, 100).is_err()); // 1 class
        assert!(train(&[vec![1.0]], &[0.0], -0.1, 100).is_err()); // lr ≤ 0
        assert!(deserialize(&[0u8; 3]).is_err());
        assert!(deserialize(&[0u8; 100]).is_err()); // k=0
    }

    #[test]
    fn blob_roundtrip_stable() {
        let x = vec![vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let m1 = train(&x, &y, 0.1, 500).unwrap();
        let m2 = deserialize(&serialize(&m1)).unwrap();
        assert_eq!(serialize(&m1), serialize(&m2)); // deterministic
        assert_eq!(m1.weights, m2.weights);
    }
}
