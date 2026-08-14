//! AdaBoost — weighted boosting of decision stumps (classification).
//!
//! Classic SAMME: each round fits a depth-1 stump on sample weights,
//! α_t = ½·ln((1−err)/err), weights updated w ← w·exp(−α·y·h) then normalized.
//! Binary labels are mapped to {−1, +1}; predict returns sign(Σ α·h).

/// Single decision stump: feature j, threshold t → left output if x[j] <= t.
#[derive(Clone, Debug)]
pub struct Stump {
    pub feature: usize,
    pub threshold: f64,
    pub left: f64,
    pub right: f64,
}

/// AdaBoost ensemble.
#[derive(Clone, Debug)]
pub struct AdaBoostResult {
    pub stumps: Vec<Stump>,
    pub alphas: Vec<f64>,
    pub classes: Vec<f64>, // original class labels, [neg, pos]
}

fn predict_stump(stump: &Stump, features: &[f64]) -> f64 {
    if features[stump.feature] <= stump.threshold {
        stump.left
    } else {
        stump.right
    }
}

/// Fit the best weighted stump (exhaustive over features and mid-points).
fn fit_stump(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> (Stump, f64) {
    let n = x.len();
    let n_features = if n > 0 { x[0].len() } else { 0 };
    let mut best: Option<(Stump, f64)> = None;

    for j in 0..n_features {
        // candidate thresholds: midpoints between sorted unique values
        let mut vals: Vec<f64> = x.iter().map(|row| row[j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        for pair in vals.windows(2) {
            let t = 0.5 * (pair[0] + pair[1]);
            for (left, right) in [(1.0, -1.0), (-1.0, 1.0)] {
                let mut err = 0.0f64;
                for i in 0..n {
                    let pred = if x[i][j] <= t { left } else { right };
                    if pred != y[i] {
                        err += w[i];
                    }
                }
                if best.is_none() || err < best.as_ref().unwrap().1 {
                    best = Some((
                        Stump {
                            feature: j,
                            threshold: t,
                            left,
                            right,
                        },
                        err,
                    ));
                }
            }
        }
    }

    match best {
        Some((s, e)) => (s, e),
        None => (
            // degenerate: single unique value per feature — constant stump
            Stump {
                feature: 0,
                threshold: 0.0,
                left: 1.0,
                right: -1.0,
            },
            0.5,
        ),
    }
}

/// Run AdaBoost on binary-classification data.
///
/// - `x`: n_samples × n_features
/// - `y`: original labels (two distinct values)
/// - `n_estimators`: boosting rounds
pub fn train(x: &[Vec<f64>], y: &[f64], n_estimators: usize) -> AdaBoostResult {
    let n = x.len();
    assert!(n > 0, "empty dataset");
    let classes: Vec<f64> = {
        let mut c: Vec<f64> = y.to_vec();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c.dedup();
        c
    };
    assert_eq!(classes.len(), 2, "adaboost requires exactly 2 classes");

    let (cneg, cpos) = (classes[0], classes[1]);
    let yb: Vec<f64> = y
        .iter()
        .map(|&v| if v == cpos { 1.0 } else { -1.0 })
        .collect();

    let mut w = vec![1.0 / n as f64; n];
    let mut stumps = Vec::with_capacity(n_estimators);
    let mut alphas = Vec::with_capacity(n_estimators);

    for _ in 0..n_estimators {
        let (stump, err) = fit_stump(x, &yb, &w);
        // clamp error away from 0 and 1 to keep alpha finite
        let err_c = err.clamp(1e-10, 1.0 - 1e-10);
        let alpha = 0.5 * ((1.0 - err_c) / err_c).ln();

        // update weights: w_i *= exp(-alpha * y_i * h(x_i)), then normalize
        let mut z = 0.0f64;
        for i in 0..n {
            let h = predict_stump(&stump, &x[i]);
            w[i] *= (-alpha * yb[i] * h).exp();
            z += w[i];
        }
        if z > 0.0 {
            let inv = 1.0 / z;
            for v in w.iter_mut() {
                *v *= inv;
            }
        }

        stumps.push(stump);
        alphas.push(alpha);
    }

    AdaBoostResult {
        stumps,
        alphas,
        classes,
    }
}

/// Ensemble prediction: sign(Σ α·h) mapped back to original class labels.
pub fn predict(result: &AdaBoostResult, features: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for (s, a) in result.stumps.iter().zip(result.alphas.iter()) {
        acc += a * predict_stump(s, features);
    }
    if acc >= 0.0 {
        result.classes[1]
    } else {
        result.classes[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XOR-like separable data with label noise removed (deterministic).
    fn separable() -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..30 {
            for j in 0..30 {
                let a = (i as f64 - 15.0) * 0.2;
                let b = (j as f64 - 15.0) * 0.2;
                x.push(vec![a, b]);
                y.push(if a + b > 0.0 { 1.0 } else { 0.0 });
            }
        }
        (x, y)
    }

    #[test]
    fn stump_ensemble_improves_over_single_stump() {
        let (x, y) = separable();
        // diagonal half-plane: a single axis-aligned stump caps at ~75%
        // (x>0 or y>0 cuts half the positives); boosting must push it higher
        let r1 = train(&x, &y, 1);
        let acc1 = accuracy(&r1, &x, &y);
        assert!(acc1 < 0.9, "single stump should NOT separate diagonal plane");
        let r30 = train(&x, &y, 30);
        let acc30 = accuracy(&r30, &x, &y);
        assert!(
            acc30 > acc1 + 0.05 && acc30 >= 0.9,
            "boosting must improve: {acc1} -> {acc30}"
        );
        assert_eq!(r30.classes, vec![0.0, 1.0]);
    }

    fn accuracy(r: &AdaBoostResult, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let mut ok = 0;
        for i in 0..x.len() {
            if predict(r, &x[i]) == y[i] {
                ok += 1;
            }
        }
        ok as f64 / x.len() as f64
    }

    #[test]
    fn alphas_are_positive() {
        let (x, y) = separable();
        let r = train(&x, &y, 5);
        for a in &r.alphas {
            assert!(*a > 0.0);
        }
    }

    #[test]
    fn predict_maps_to_original_labels() {
        let (x, y) = separable();
        let r = train(&x, &y, 3);
        for &v in &y {
            assert!(r.classes.contains(&v));
        }
        for i in 0..x.len() {
            let p = predict(&r, &x[i]);
            assert!(p == 0.0 || p == 1.0);
        }
    }
}
