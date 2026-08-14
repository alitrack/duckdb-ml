//! Polynomial regression — feature expansion (per-feature powers 1..=degree)
//! delegated to the OLS/Ridge core in train/linear.rs.
//!
//! No cross-feature interaction terms (documented): predictions match sklearn
//! PolynomialFeatures + LinearRegression exactly for single-feature targets
//! (no interactions possible); multi-feature targets share the linear core
//! so predictive quality holds while coefficient sets differ.

use crate::train::TrainingResult;

/// Expand (n × d) into (n × d·degree): for each feature x_j, columns
/// x_j^1, x_j^2, …, x_j^degree. Caller (table_fn) unwraps the coefficients
/// with the same ordering.
pub fn expand_features(x: &[Vec<f64>], degree: usize) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let mut out = Vec::with_capacity(row.len() * degree);
            for &v in row {
                let mut p = v;
                for _ in 1..=degree {
                    out.push(p);
                    p *= v;
                }
            }
            out
        })
        .collect()
}

/// Fit a polynomial regression by expanding features and delegating to OLS/Ridge.
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    degree: usize,
    lambda: f64,
) -> Result<TrainingResult, Box<dyn std::error::Error>> {
    if degree == 0 {
        return Err("polynomial_regression: degree must be >= 1".into());
    }
    let expanded = expand_features(x, degree);
    super::linear::train(&expanded, y, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_shape_and_values() {
        let x = vec![vec![2.0, 3.0]];
        let e = expand_features(&x, 3);
        assert_eq!(e[0], vec![2.0, 4.0, 8.0, 3.0, 9.0, 27.0]);
    }

    #[test]
    fn fits_quadratic_exactly() {
        // y = 1 + 2x − 3x²  on single feature
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 / 2.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| 1.0 + 2.0 * r[0] - 3.0 * r[0] * r[0]).collect();
        let r = train(&x, &y, 2, 0.0).unwrap();
        // coefficients = [x^1, x^2, intercept] → [2, -3, 1]
        assert!((r.coefficients[0] - 2.0).abs() < 1e-8, "{:?}", r.coefficients);
        assert!((r.coefficients[1] + 3.0).abs() < 1e-8, "{:?}", r.coefficients);
        assert!((r.coefficients[2] - 1.0).abs() < 1e-8, "{:?}", r.coefficients);
    }
}
