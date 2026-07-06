use super::TrainingResult;
use std::error::Error;

/// Train OLS or Ridge linear regression
/// Uses Gaussian elimination for solving the normal equations
pub fn train(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Result<TrainingResult, Box<dyn Error>> {
    let n_samples = x.len();
    let n_features = x[0].len();
    let n_cols = n_features + 1; // + intercept

    // Build X^T X matrix (n_cols × n_cols) and X^T y vector
    let mut xtx = vec![0.0f64; n_cols * n_cols];
    let mut xty = vec![0.0f64; n_cols];

    for i in 0..n_samples {
        for j in 0..n_cols {
            let xj = if j < n_features { x[i][j] } else { 1.0 };
            xty[j] += xj * y[i];
            for k in 0..n_cols {
                let xk = if k < n_features { x[i][k] } else { 1.0 };
                xtx[j * n_cols + k] += xj * xk;
            }
        }
    }

    // Ridge: add λ to diagonal (except intercept)
    for j in 0..n_features {
        xtx[j * n_cols + j] += lambda;
    }

    // Solve via Gaussian elimination with partial pivoting
    let mut a = xtx;
    let mut b = xty;
    let n = n_cols;

    for col in 0..n {
        // Partial pivot
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err("Matrix is singular or nearly singular".into());
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }

        // Eliminate below
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut coeffs = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * coeffs[j];
        }
        coeffs[i] = sum / a[i * n + i];
    }

    let mut coefficients = Vec::with_capacity(n_cols);
    for &c in coeffs.iter().take(n_features) {
        coefficients.push(c);
    }
    let intercept = coeffs[n_features];
    coefficients.push(intercept);

    // Metrics
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
    let model_intercept = coefficients[n_features];

    Ok(TrainingResult {
        coefficients,
        intercept: model_intercept,
        r_squared,
        mse,
        num_samples: n_samples,
    })
}
