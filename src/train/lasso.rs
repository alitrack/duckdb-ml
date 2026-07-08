//! Lasso Regression Training — Coordinate Descent with Soft-Thresholding
//!
//! Minimizes: (1/2n) * ||y - Xβ||² + λ * ||β||₁
//!
//! Soft-thresholding operator: S(z, γ) = sign(z) * max(|z| - γ, 0)

/// Result type for Lasso training
pub type LassoResult = (Vec<f64>, Option<f64>, Option<f64>);

/// Train a Lasso model using cyclic coordinate descent.
///
/// Returns: (coefficients including intercept at the end, r_squared, mse)
pub fn train_lasso(
    x: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<LassoResult, String> {
    let n = x.len();
    let p = x[0].len();

    if n < 2 {
        return Err("Need at least 2 samples".into());
    }
    if !x.iter().all(|row| row.len() == p) {
        return Err("Features rows have inconsistent column counts".into());
    }

    // Standardize X and center y (Lasso is not scale-invariant)
    let (xs, x_mean, x_std) = standardize(x);
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();

    // Initialize coefficients to zero
    let mut beta = vec![0.0f64; p];

    // Pre-compute X^T X diagonals (after standardization, diag_i = n)
    // Pre-compute X^T y (correlations)
    let mut xty = vec![0.0f64; p];
    for j in 0..p {
        for i in 0..n {
            xty[j] += xs[i][j] * y_centered[i];
        }
    }

    let n_f = n as f64;
    let lambda_scaled = lambda * n_f; // scale by n for coordinate descent

    for iter in 0..max_iter {
        let beta_old = beta.clone();
        let mut max_change = 0.0f64;

        for j in 0..p {
            // Compute residual excluding feature j
            let mut rho = xty[j];
            for (k, beta_k) in beta.iter().enumerate().take(p) {
                if k != j {
                    let mut xk_dot_xj = 0.0;
                    for row in xs.iter().take(n) {
                        xk_dot_xj += row[k] * row[j];
                    }
                    rho -= xk_dot_xj * beta_k;
                }
            }

            // ||x_j||² = n after standardization
            let xj_norm2 = n_f;

            // Soft-threshold
            let z = rho / xj_norm2;
            if rho > lambda_scaled {
                beta[j] = z - lambda_scaled / xj_norm2;
            } else if rho < -lambda_scaled {
                beta[j] = z + lambda_scaled / xj_norm2;
            } else {
                beta[j] = 0.0;
            }

            let change = (beta[j] - beta_old[j]).abs();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }
        if iter == max_iter - 1 {
            // Converged in time, no warning needed
        }
    }

    // Un-standardize: β_original = β_standardized / x_std
    let mut coef_unscaled = vec![0.0f64; p];
    for j in 0..p {
        coef_unscaled[j] = beta[j] / x_std[j];
    }

    // Compute intercept: β₀ = y_mean - Σ β_j * x_mean_j
    let intercept = y_mean
        - coef_unscaled
            .iter()
            .zip(x_mean.iter())
            .map(|(b, m)| b * m)
            .sum::<f64>();

    let mut final_coef = coef_unscaled;
    final_coef.push(intercept);

    // Compute metrics
    let predictions: Vec<f64> = x
        .iter()
        .map(|xi| {
            let mut pred = intercept;
            for (j, &v) in xi.iter().enumerate() {
                pred += final_coef[j] * v;
            }
            pred
        })
        .collect();

    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(y.iter())
        .map(|(&p, &yi)| (yi - p).powi(2))
        .sum();
    let r2 = if ss_tot > 0.0 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    };
    let mse = Some(ss_res / n_f);

    Ok((final_coef, r2, mse))
}

/// Standardize features to zero mean and unit variance
fn standardize(x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    let p = x[0].len();

    let mut mean = vec![0.0f64; p];
    for row in x {
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }

    let mut std = vec![0.0f64; p];
    for row in x {
        for (s, (&v, &m)) in std.iter_mut().zip(row.iter().zip(mean.iter())) {
            *s += (v - m).powi(2);
        }
    }
    for s in std.iter_mut() {
        *s = (*s / n as f64).sqrt();
        if *s < 1e-10 {
            *s = 1.0;
        }
    }

    let xs: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .zip(mean.iter().zip(std.iter()))
                .map(|(&v, (&m, &s))| (v - m) / s)
                .collect()
        })
        .collect();

    (xs, mean, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasso_simple() {
        // y = 3*x1 + 2*x2 + 1
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
            vec![6.0, 5.0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 2.0 * xi[1] + 1.0).collect();

        let (coef, r2, _mse) = train_lasso(&x, &y, 0.01, 1000, 1e-4).unwrap();
        assert_eq!(coef.len(), 3); // coeff1, coeff2, intercept
        assert!(r2.unwrap() > 0.9, "r2 too low: {:?}", r2);
    }

    #[test]
    fn test_lasso_sparsity() {
        // y = 3*x1 + 0*x2 + 1 (x2 is irrelevant)
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 0.3],
            vec![3.0, 0.9],
            vec![4.0, 0.2],
            vec![5.0, 0.7],
            vec![6.0, 0.4],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 1.0).collect();

        // With strong lambda, irrelevant feature should shrink toward zero
        let (coef, _, _) = train_lasso(&x, &y, 5.0, 1000, 1e-4).unwrap();
        // x1 coefficient should be dominant, x2 should be small or zero
        assert!(
            coef[0].abs() > coef[1].abs() * 2.0,
            "x1={} should dominate x2={}",
            coef[0],
            coef[1]
        );
    }
}
