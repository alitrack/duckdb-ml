//! Ordinal logistic regression (cumulative logit / proportional odds) —
//! MADlib `ordinal` counterpart.
//!
//! Model: P(Y ≤ j | x) = σ(θ_j − w·x) with ordered thresholds θ_1 < … < θ_{K-1}
//! shared weight vector w. Class probabilities: p_1 = c_1, p_j = c_j − c_{j−1},
//! p_K = 1 − c_{K−1}. Trained by full-batch gradient descent on negative
//! log-likelihood; thresholds are re-parameterized θ_j = θ_1 + Σ_{m=2..j} e^{δ_m}
//! so monotonicity is guaranteed by construction. Deterministic (0-init).
//!
//! Blob layout (little-endian): u32 k · u32 d · k×f64 classes ·
//! d×f64 weights · (k−1)×f64 thresholds

/// Trained ordinal logistic model.
pub struct OrdinalModel {
    /// Unique sorted class labels (original values).
    pub classes: Vec<f64>,
    /// Shared weight vector (length d).
    pub weights: Vec<f64>,
    /// Increasing thresholds θ_1..θ_{K-1} (length k−1).
    pub thresholds: Vec<f64>,
    pub n_features: usize,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Train an ordinal logistic model. `y` must be numeric ordinal labels
/// (e.g. 0/1/2 or 1/2/3); at least 2 distinct classes.
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_epochs: usize,
) -> Result<OrdinalModel, String> {
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

    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();
    let k = classes.len();
    if k < 2 {
        return Err(format!("need ≥ 2 classes, got {k}"));
    }
    let y_idx: Vec<usize> = y
        .iter()
        .map(|v| {
            classes
                .iter()
                .position(|c| (c - v).abs() < 1e-12)
                .expect("class in set")
        })
        .collect();

    let nt = k - 1; // number of thresholds
                    // free params: weights (d) + theta1 + deltas (nt−1); deltas → exp() → positive
    let mut w = vec![0.0f64; d];
    let mut theta1 = 0.0f64;
    let mut deltas = vec![0.0f64; nt - 1]; // delta_m = log(θ_{m+1} − θ_m)

    let mut prev_loss = f64::MAX;
    for _epoch in 0..max_epochs {
        // thresholds from re-parameterization
        let mut th = vec![0.0f64; nt];
        th[0] = theta1;
        for m in 1..nt {
            th[m] = th[m - 1] + deltas[m - 1].exp();
        }

        let mut grad_w = vec![0.0f64; d];
        let mut grad_t1 = 0.0f64;
        let mut grad_delta = vec![0.0f64; nt - 1];
        let mut grad_theta = vec![0.0f64; nt];
        let mut total_loss = 0.0f64;

        for i in 0..n {
            let dot = w.iter().zip(&x[i]).map(|(a, b)| a * b).sum::<f64>();
            // cumulative probabilities c_j = σ(θ_j − dot)
            let mut c = vec![0.0f64; nt];
            for j in 0..nt {
                c[j] = sigmoid(th[j] - dot);
            }
            // class probabilities
            let mut p = vec![0.0f64; k];
            p[0] = c[0];
            for j in 1..nt {
                p[j] = (c[j] - c[j - 1]).max(1e-12);
            }
            p[k - 1] = (1.0 - c[nt - 1]).max(1e-12);

            let yi = y_idx[i];
            total_loss += -p[yi].ln();
            let inv_p = 1.0 / p[yi];

            // exact NLL gradients:
            //   ∂LL/∂θ_j = (1/p_yi)·c_j(1−c_j)·[δ(yi=j) − δ(yi=j+1)]
            //   ∂LL/∂w    = −x·Σ_j ∂LL/∂θ_j   (since z_j = θ_j − w·x)
            let mut sum_gt = 0.0f64;
            for j in 0..nt {
                let ind = if yi == j {
                    1.0
                } else if yi == j + 1 {
                    -1.0
                } else {
                    0.0
                };
                let g = inv_p * c[j] * (1.0 - c[j]) * ind;
                grad_theta[j] += g;
                sum_gt += g;
            }
            for (a, b) in grad_w.iter_mut().zip(&x[i]) {
                *a += -b * sum_gt;
            }
        }

        // chain θ gradients to free params: θ_j = θ_1 + Σ_{m=1..j-1} e^{δ_m}
        grad_t1 += grad_theta.iter().sum::<f64>();
        for m in 0..nt - 1 {
            let s = grad_theta[m + 1..].iter().sum::<f64>();
            grad_delta[m] += s * deltas[m].exp();
        }

        total_loss /= n as f64;
        for a in grad_w.iter_mut() {
            *a /= n as f64;
        }
        grad_t1 /= n as f64;
        for a in grad_delta.iter_mut() {
            *a /= n as f64;
        }

        for j in 0..d {
            w[j] += lr * grad_w[j];
        }
        theta1 += lr * grad_t1;
        for m in 0..nt - 1 {
            deltas[m] += lr * grad_delta[m];
        }

        if (prev_loss - total_loss).abs() < 1e-7 {
            break;
        }
        prev_loss = total_loss;
    }

    let mut thresholds = vec![0.0f64; nt];
    thresholds[0] = theta1;
    for m in 1..nt {
        thresholds[m] = thresholds[m - 1] + deltas[m - 1].exp();
    }

    Ok(OrdinalModel {
        classes,
        weights: w,
        thresholds,
        n_features: d,
    })
}

/// Predict the most probable class label for one feature row.
pub fn predict_one(m: &OrdinalModel, features: &[f64]) -> f64 {
    let dot = m
        .weights
        .iter()
        .zip(features)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let nt = m.thresholds.len();
    let mut p = vec![0.0f64; nt + 1];
    let mut c_prev = 0.0f64;
    for (j, pj) in p[..nt].iter_mut().enumerate() {
        let c = sigmoid(m.thresholds[j] - dot);
        *pj = (c - c_prev).max(0.0);
        c_prev = c;
    }
    p[nt] = (1.0 - c_prev).max(0.0);
    let mut best = 0usize;
    for j in 1..p.len() {
        if p[j] > p[best] {
            best = j;
        }
    }
    m.classes[best]
}

pub fn serialize(m: &OrdinalModel) -> Vec<u8> {
    let k = m.classes.len();
    let d = m.n_features;
    let mut out = Vec::with_capacity(8 + (k + d + k - 1) * 8);
    out.extend_from_slice(&(k as u32).to_le_bytes());
    out.extend_from_slice(&(d as u32).to_le_bytes());
    for c in &m.classes {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for w in &m.weights {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for t in &m.thresholds {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

pub fn deserialize(blob: &[u8]) -> Result<OrdinalModel, String> {
    if blob.len() < 8 {
        return Err("truncated blob".into());
    }
    let k = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(blob[4..8].try_into().unwrap()) as usize;
    if k < 2 || d == 0 || k > 100_000 {
        return Err("implausible k/d".into());
    }
    let need = 8 + (k + d + k - 1) * 8;
    if blob.len() < need {
        return Err("truncated blob".into());
    }
    let rd = |off: usize| -> f64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&blob[off..off + 8]);
        f64::from_le_bytes(b)
    };
    let mut classes = Vec::with_capacity(k);
    for i in 0..k {
        classes.push(rd(8 + i * 8));
    }
    let base_w = 8 + k * 8;
    let mut weights = Vec::with_capacity(d);
    for i in 0..d {
        weights.push(rd(base_w + i * 8));
    }
    let base_t = base_w + d * 8;
    let mut thresholds = Vec::with_capacity(k - 1);
    for i in 0..k - 1 {
        thresholds.push(rd(base_t + i * 8));
    }
    Ok(OrdinalModel {
        classes,
        weights,
        thresholds,
        n_features: d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // y grows with x: low x → class 0, mid → 1, high → 2
    fn ordinal_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..15 {
            let t = i as f64;
            x.push(vec![t * 0.5]);
            x.push(vec![t * 0.5 + 1.0]);
            let cls = if t < 5.0 {
                0.0
            } else if t < 10.0 {
                1.0
            } else {
                2.0
            };
            y.push(cls);
            y.push(cls);
        }
        (x, y)
    }

    #[test]
    fn three_levels_ordered() {
        let (x, y) = ordinal_data();
        let m = train(&x, &y, 0.1, 2000).unwrap();
        assert_eq!(m.classes, vec![0.0, 1.0, 2.0]);
        // monotone ordering respected
        assert!(m.thresholds[0] < m.thresholds[1]);
        assert_eq!(predict_one(&m, &[0.0]), 0.0);
        assert_eq!(predict_one(&m, &[2.0]), 0.0); // t=4 → class 0
        assert_eq!(predict_one(&m, &[3.0]), 1.0); // t=6 → class 1
        assert_eq!(predict_one(&m, &[4.5]), 1.0); // t=9 → class 1
        assert_eq!(predict_one(&m, &[12.0]), 2.0); // t≥10 → class 2
    }

    #[test]
    fn non_consecutive_labels() {
        let x = vec![vec![-3.0], vec![-2.0], vec![0.0], vec![3.0]];
        let y = vec![1.0, 1.0, 2.0, 3.0];
        let m = train(&x, &y, 0.1, 1500).unwrap();
        assert_eq!(m.classes, vec![1.0, 2.0, 3.0]);
        assert_eq!(predict_one(&m, &[-3.0]), 1.0);
        assert_eq!(predict_one(&m, &[3.0]), 3.0);
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[], &[], 0.1, 100).is_err());
        assert!(train(&[vec![1.0]], &[0.0], 0.1, 100).is_err()); // 1 class
        assert!(train(&[vec![1.0]], &[0.0], -0.1, 100).is_err());
        assert!(deserialize(&[0u8; 7]).is_err());
    }

    #[test]
    fn roundtrip_stable() {
        let (x, y) = ordinal_data();
        let m1 = train(&x, &y, 0.1, 500).unwrap();
        let m2 = deserialize(&serialize(&m1)).unwrap();
        assert_eq!(serialize(&m1), serialize(&m2));
        assert_eq!(predict_one(&m2, &[5.0]), predict_one(&m1, &[5.0]));
    }
}
