//! Lasso Regression (L1-regularized linear regression)
//!
//! Coordinate descent with soft-thresholding.
//! β_j = S(ρ_j, λ) / ||x_j||²  where ρ_j = x_j^T (y - X_{-j}β_{-j})
//! and S(z, γ) = sign(z) * max(|z| - γ, 0)

use super::types::{Algorithm, MlModel, ModelError, ModelMetadata};

/// Core Lasso model — coefficients only, prediction = dot + intercept
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct LassoCore {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub lambda: f64,
}

/// MlModel wrapper
pub struct LassoModel {
    pub metadata: ModelMetadata,
    pub inner: LassoCore,
}

impl LassoModel {
    pub fn new(
        coefficients: Vec<f64>,
        num_samples: usize,
        r_squared: Option<f64>,
        mse: Option<f64>,
        lambda: f64,
    ) -> Self {
        let num_features = coefficients.len().saturating_sub(1); // last is intercept
        Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::LassoRegression,
                num_features,
                num_samples,
                r_squared,
                mse,
                coefficients_count: coefficients.len(),
                hyperparameters_json: format!(r#"{{"lambda":{lambda}}}"#),
            },
            inner: LassoCore {
                intercept: coefficients.last().copied().unwrap_or(0.0),
                coefficients: coefficients[..coefficients.len().saturating_sub(1)].to_vec(),
                lambda,
            },
        }
    }
}

impl MlModel for LassoModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        let mut pred = self.inner.intercept;
        for (i, &x) in features.iter().enumerate() {
            pred += self.inner.coefficients[i] * x;
        }
        Ok(pred)
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::LassoRegression
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        bincode::encode_to_vec(&self.inner, bincode::config::standard())
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError> {
        let (core, _): (LassoCore, _) =
            bincode::decode_from_slice(blob, bincode::config::standard())
                .map_err(|e| ModelError::Serialization(e.to_string()))?;
        let num_features = core.coefficients.len();
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::LassoRegression,
                num_features,
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: core.coefficients.len() + 1,
                hyperparameters_json: format!(r#"{{"lambda":{}}}"#, core.lambda),
            },
            inner: core,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasso_predict() {
        let model = LassoModel::new(
            vec![3.0, 2.0, 1.0], // coeffs + intercept
            6,
            Some(0.95),
            Some(0.5),
            0.01,
        );
        // y = 3*x1 + 2*x2 + 1
        let pred = model.predict(&[1.0, 2.0]).unwrap();
        assert!((pred - 8.0).abs() < 0.01, "got {pred}");
    }

    #[test]
    fn test_lasso_serialization() {
        let model = LassoModel::new(vec![3.0, 2.0, 1.0], 6, Some(0.95), Some(0.5), 0.01);
        let blob = model.serialize().unwrap();
        let restored = LassoModel::deserialize(&blob).unwrap();
        let pred = restored.predict(&[1.0, 2.0]).unwrap();
        assert!((pred - 8.0).abs() < 0.01);
    }
}
