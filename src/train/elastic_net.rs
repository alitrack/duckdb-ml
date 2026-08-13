//! Elastic Net regression — hand-written coordinate descent (PostgresML's
//! smartcore-backed path was evaluated and rejected: its interior-point
//! optimizer is unreliable on both normalize paths).
//!
//! Elastic net = ridge + lasso blend: minimizes
//!   (1/2n)·||y − Xβ||² + α·l1_ratio·||β||₁ + (α/2)·(1−l1_ratio)·||β||₂²
//! via cyclical coordinate descent with soft-thresholding (sklearn's
//! standard algorithm). Deterministic, zero dependencies. Coefficients are
//! on the raw-feature scale; intercept last (reuses `linear_regression`
//! blob format).

use super::TrainingResult;
use std::error::Error;

/// Train elastic-net regression.
///
/// `alpha` = overall regularization strength (default 1.0),
/// `l1_ratio` = mixing weight for L1 (default 0.5; 0 = pure ridge,
/// 1 = pure lasso), `max_iter` = coordinate-descent cap (default 1000).
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
) -> Result<TrainingResult, Box<dyn Error>> {
    let n_samples = x.len();
    if n_samples == 0 {
        return Err("empty training data".into());
    }
    if alpha < 0.0 {
        return Err("alpha must be >= 0".into());
    }
    if !(0.0..=1.0).contains(&l1_ratio) {
        return Err("l1_ratio must be in [0, 1]".into());
    }
    let n_features = x[0].len();
    if x.iter().any(|s| s.len() != n_features) {
        return Err("inconsistent feature dimensions".into());
    }
    if y.len() != n_samples {
        return Err("x/y length mismatch".into());
    }

    // center y and X (coordinate descent assumes zero-mean columns);
    // intercept is recovered analytically afterwards
    let y_mean = y.iter().sum::<f64>() / n_samples as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - y_mean).collect();
    let x_means: Vec<f64> = (0..n_features)
        .map(|j| x.iter().map(|s| s[j]).sum::<f64>() / n_samples as f64)
        .collect();
    let xc: Vec<Vec<f64>> = x
        .iter()
        .map(|s| s.iter().zip(&x_means).map(|(v, m)| v - m).collect())
        .collect();
    let mut x_col_sq = vec![0.0f64; n_features];
    for row in &xc {
        for j in 0..n_features {
            x_col_sq[j] += row[j] * row[j];
        }
    }

    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio);
    let mut beta = vec![0.0f64; n_features];

    for _ in 0..max_iter {
        let mut max_delta = 0.0f64;
        for j in 0..n_features {
            let old = beta[j];
            // ρ_j = Σ_i x_ij·(yc_i − ŷ_i) + β_j·||x_j||²  (residual trick)
            let mut rho = 0.0f64;
            for i in 0..n_samples {
                let mut pred = 0.0f64;
                for k in 0..n_features {
                    pred += beta[k] * xc[i][k];
                }
                rho += xc[i][j] * (yc[i] - pred);
            }
            rho += old * x_col_sq[j];
            let denom = x_col_sq[j] + l2 * n_samples as f64;
            if denom <= 1e-12 {
                beta[j] = 0.0;
                continue;
            }
            let z = rho / (n_samples as f64);
            // soft threshold: β_j = soft(z, l1)·n / denom
            let new = if z > l1 {
                (z - l1) / denom * n_samples as f64
            } else if z < -l1 {
                (z + l1) / denom * n_samples as f64
            } else {
                0.0
            };
            beta[j] = new;
            let d = (new - old).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        if max_delta < 1e-8 {
            break;
        }
    }

    // intercept: b = y_mean − Σ_j β_j·x̄_j
    let mut intercept = y_mean;
    for j in 0..n_features {
        intercept -= beta[j] * x_means[j];
    }

    let mut coefficients = beta;
    coefficients.push(intercept);

    // Metrics on the final fit
    let mut predictions = Vec::with_capacity(n_samples);
    for sample in x.iter() {
        let mut pred = intercept;
        for j in 0..n_features {
            pred += coefficients[j] * sample[j];
        }
        predictions.push(pred);
    }
    let ss_res: f64 = predictions
        .iter()
        .zip(y.iter())
        .map(|(p, a)| (a - p).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|a| (a - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 1e-10 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    };
    let mse = Some(ss_res / n_samples as f64);

    Ok(TrainingResult {
        coefficients,
        intercept,
        r_squared,
        mse,
        num_samples: n_samples,
        model_blob: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_line_fit() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| 2.0 + 3.0 * i as f64).collect();
        let r = train(&x, &y, 1e-6, 0.5, 5000).unwrap();
        assert!(
            (r.coefficients[0] - 3.0).abs() < 1e-4,
            "slope={}",
            r.coefficients[0]
        );
        assert!(
            (r.intercept - 2.0).abs() < 1e-4,
            "intercept={}",
            r.intercept
        );
    }

    #[test]
    fn ridge_limit_recovers_ols() {
        // l1_ratio=0 → pure ridge; tiny alpha ≈ OLS
        let x: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64 * 0.7]).collect();
        let y: Vec<f64> = (0..12).map(|i| 1.0 + 2.0 * i as f64 * 0.7).collect();
        let r = train(&x, &y, 1e-6, 0.0, 5000).unwrap();
        assert!(
            (r.coefficients[0] - 2.0).abs() < 1e-3,
            "slope={}",
            r.coefficients[0]
        );
        assert!(
            (r.intercept - 1.0).abs() < 1e-3,
            "intercept={}",
            r.intercept
        );
    }

    #[test]
    fn lasso_like_sparsity() {
        // l1_ratio=1 → lasso path; irrelevant feature shrinks toward 0
        let mut x: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for i in 0..30 {
            let v = i as f64 * 0.5;
            x.push(vec![v, (v * 3.7).sin()]); // x2 oscillates, unrelated to y
            y.push(3.0 * v + 0.5);
        }
        let r = train(&x, &y, 0.3, 1.0, 3000).unwrap();
        assert!(
            (r.coefficients[0] - 3.0).abs() < 0.2,
            "b1={}",
            r.coefficients[0]
        );
        assert!(
            r.coefficients[1].abs() < 0.5,
            "b2={} (should be shrunk)",
            r.coefficients[1]
        );
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[], &[], 1.0, 0.5, 1000).is_err());
        assert!(train(&[vec![1.0]], &[1.0], -1.0, 0.5, 1000).is_err());
        assert!(train(&[vec![1.0]], &[1.0], 1.0, 1.5, 1000).is_err());
        assert!(train(&[vec![1.0, 2.0], vec![1.0]], &[1.0, 2.0], 1.0, 0.5, 1000).is_err());
        assert!(train(&[vec![1.0]], &[1.0, 2.0], 1.0, 0.5, 1000).is_err());
    }

    #[test]
    fn deterministic() {
        let x: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..15).map(|i| 1.0 + 2.0 * i as f64).collect();
        let a = train(&x, &y, 0.01, 0.5, 1000).unwrap();
        let b = train(&x, &y, 0.01, 0.5, 1000).unwrap();
        assert_eq!(a.coefficients, b.coefficients);
    }
}
