//! Robust linear regression (Huber loss via iteratively reweighted least
//! squares) — MADlib `robust` counterpart.
//!
//! IRLS: start from OLS, then reweight by Huber weights
//!   u_i = r_i / (1.4826·MAD(r)),  w_i = 1 if |u_i| ≤ c else c/|u_i|
//! and solve the weighted normal equations (X'WX)β = X'Wy until β converges.
//! Down-weights outliers, so the fit is resistant to leverage points.
//! Deterministic (no randomness). Returns coefficients (intercept last),
//! matching the `linear_regression` model format.

use super::TrainingResult;
use std::error::Error;

/// Train robust (Huber) linear regression.
///
/// `c` = Huber cutoff in scaled-residual units (default 1.345),
/// `max_iters` = IRLS iteration cap (default 50).
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    c: f64,
    max_iters: usize,
) -> Result<TrainingResult, Box<dyn Error>> {
    let n_samples = x.len();
    if n_samples == 0 {
        return Err("robust: empty training data".into());
    }
    if c <= 0.0 {
        return Err("robust: c must be > 0".into());
    }
    let n_features = x[0].len();
    if x.iter().any(|s| s.len() != n_features) {
        return Err("robust: inconsistent feature dimensions".into());
    }
    if y.len() != n_samples {
        return Err("robust: x/y length mismatch".into());
    }
    let n_cols = n_features + 1; // + intercept

    // initial OLS coefficients
    let mut beta = solve_weighted(x, y, &vec![1.0f64; n_samples], n_features, n_cols)?;

    for _iter in 0..max_iters {
        // residuals
        let mut res = vec![0.0f64; n_samples];
        for i in 0..n_samples {
            let mut pred = beta[n_features];
            for j in 0..n_features {
                pred += beta[j] * x[i][j];
            }
            res[i] = y[i] - pred;
        }
        // robust scale: 1.4826 · MAD
        let mut sorted = res.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let mut abs_dev: Vec<f64> = sorted.iter().map(|r| (r - median).abs()).collect();
        abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_dev[abs_dev.len() / 2];
        let scale = (1.4826 * mad).max(1e-9);

        // Huber weights
        let mut w = vec![0.0f64; n_samples];
        for i in 0..n_samples {
            let u = res[i] / scale;
            w[i] = if u.abs() <= c { 1.0 } else { c / u.abs() };
        }

        let new_beta = solve_weighted(x, y, &w, n_features, n_cols)?;
        let delta: f64 = beta
            .iter()
            .zip(&new_beta)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        beta = new_beta;
        if delta < 1e-8 {
            break;
        }
    }

    let mut coefficients = beta[..n_features].to_vec();
    let intercept = beta[n_features];
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
    let y_mean = y.iter().sum::<f64>() / n_samples as f64;
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

/// Solve (X'WX)β = X'Wy via Gaussian elimination with partial pivoting.
/// X rows are [x_i, 1] (intercept column last).
fn solve_weighted(
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    n_features: usize,
    n_cols: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut xtx = vec![0.0f64; n_cols * n_cols];
    let mut xty = vec![0.0f64; n_cols];
    for i in 0..x.len() {
        let wi = w[i];
        for j in 0..n_cols {
            let xj = if j < n_features { x[i][j] } else { 1.0 };
            xty[j] += wi * xj * y[i];
            for k in 0..n_cols {
                let xk = if k < n_features { x[i][k] } else { 1.0 };
                xtx[j * n_cols + k] += wi * xj * xk;
            }
        }
    }

    let mut a = xtx;
    let mut b = xty;
    for col in 0..n_cols {
        let mut piv = col;
        for r in col + 1..n_cols {
            if a[r * n_cols + col].abs() > a[piv * n_cols + col].abs() {
                piv = r;
            }
        }
        if a[piv * n_cols + col].abs() < 1e-12 {
            return Err("robust: singular design matrix".into());
        }
        if piv != col {
            for k in 0..n_cols {
                a.swap(piv * n_cols + k, col * n_cols + k);
            }
            b.swap(piv, col);
        }
        for r in col + 1..n_cols {
            let f = a[r * n_cols + col] / a[col * n_cols + col];
            if f != 0.0 {
                for k in col..n_cols {
                    a[r * n_cols + k] -= f * a[col * n_cols + k];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut beta = vec![0.0f64; n_cols];
    for col in (0..n_cols).rev() {
        let mut s = b[col];
        for k in col + 1..n_cols {
            s -= a[col * n_cols + k] * beta[k];
        }
        beta[col] = s / a[col * n_cols + col];
    }
    Ok(beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_line_fit() {
        // y = 2 + 3x, no noise
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| 2.0 + 3.0 * i as f64).collect();
        let r = train(&x, &y, 1.345, 50).unwrap();
        assert!(
            (r.coefficients[0] - 3.0).abs() < 1e-6,
            "slope={}",
            r.coefficients[0]
        );
        assert!(
            (r.intercept - 2.0).abs() < 1e-6,
            "intercept={}",
            r.intercept
        );
        assert_eq!(r.coefficients.len(), 2);
    }

    #[test]
    fn outlier_downweighted() {
        // y = 3x with one gross outlier (should be mostly ignored)
        let mut x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let mut y: Vec<f64> = (0..20).map(|i| 3.0 * i as f64).collect();
        x[10] = vec![10.0];
        y[10] = 1000.0; // outlier
        let r = train(&x, &y, 1.345, 50).unwrap();
        // OLS would be pulled up; robust slope stays near 3
        assert!(
            (r.coefficients[0] - 3.0).abs() < 0.3,
            "slope={}",
            r.coefficients[0]
        );
        assert!(r.intercept.abs() < 3.0, "intercept={}", r.intercept);
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[], &[], 1.345, 50).is_err());
        assert!(train(&[vec![1.0]], &[1.0], 0.0, 50).is_err());
        assert!(train(&[vec![1.0, 2.0], vec![1.0]], &[1.0, 2.0], 1.345, 50).is_err());
        assert!(train(&[vec![1.0]], &[1.0, 2.0], 1.345, 50).is_err());
    }

    #[test]
    fn deterministic() {
        let x: Vec<Vec<f64>> = (0..15)
            .map(|i| vec![i as f64, i as f64 * i as f64 * 0.1])
            .collect();
        let y: Vec<f64> = (0..15).map(|i| 1.0 + 2.0 * i as f64).collect();
        let a = train(&x, &y, 1.345, 50).unwrap();
        let b = train(&x, &y, 1.345, 50).unwrap();
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.intercept, b.intercept);
    }

    #[test]
    fn multiple_features() {
        // y = 1 + 2x1 - 3x2
        let x: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![i as f64 * 0.7, i as f64 * i as f64 * 0.1])
            .collect();
        let y: Vec<f64> = (0..12)
            .map(|i| 1.0 + 2.0 * i as f64 * 0.7 - 3.0 * i as f64 * i as f64 * 0.1)
            .collect();
        let r = train(&x, &y, 1.345, 50).unwrap();
        assert!(
            (r.coefficients[0] - 2.0).abs() < 1e-6,
            "b1={}",
            r.coefficients[0]
        );
        assert!(
            (r.coefficients[1] + 3.0).abs() < 1e-6,
            "b2={}",
            r.coefficients[1]
        );
        assert!(
            (r.intercept - 1.0).abs() < 1e-6,
            "intercept={}",
            r.intercept
        );
    }
}
