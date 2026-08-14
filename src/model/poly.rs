//! PolyModel — MlModel wrapper for polynomial regression.
//!
//! Stores the expanded-feature LinearModel plus the degree so predict() can
//! re-expand raw features. Coefficients are ordered [x1^1, x1^2, …, xd^degree,
//! intercept] matching train/poly.rs expansion.

use super::linear::LinearModel;
use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct PolyModel {
    pub metadata: ModelMetadata,
    inner: LinearModel,
    degree: usize,
    n_features: usize,
}

impl PolyModel {
    pub fn new(
        coefficients: Vec<f64>,
        degree: usize,
        num_samples: usize,
        r_squared: Option<f64>,
        mse: Option<f64>,
        lambda: f64,
    ) -> Self {
        // coefficients = d·degree powers + intercept
        let n_features = (coefficients.len() - 1) / degree;
        let inner = LinearModel::new(coefficients, num_samples, r_squared, mse, lambda);
        let metadata = ModelMetadata {
            algorithm: Algorithm::PolynomialRegression,
            num_features: n_features,
            num_samples,
            r_squared,
            mse,
            coefficients_count: inner.coefficients().len(),
            hyperparameters_json: serde_json::json!({
                "degree": degree,
                "lambda": lambda,
                "feature_order": "per-feature powers x^1..x^degree",
            })
            .to_string(),
        };
        Self {
            metadata,
            inner,
            degree,
            n_features,
        }
    }

    fn expand(&self, features: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(features.len() * self.degree);
        for &v in features {
            let mut p = v;
            for _ in 1..=self.degree {
                out.push(p);
                p *= v;
            }
        }
        out
    }

    pub fn coefficients(&self) -> &[f64] {
        self.inner.coefficients()
    }

    pub fn degree(&self) -> usize {
        self.degree
    }
}

impl MlModel for PolyModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.n_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.n_features,
                got: features.len(),
            });
        }
        self.inner.predict(&self.expand(features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::PolynomialRegression
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        bincode::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError> {
        bincode::decode_from_slice(blob, bincode::config::standard())
            .map(|(m, _)| m)
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_model() -> PolyModel {
        // y = 1 + 2x − 3x² → coefficients [2, -3, 1], degree 2
        PolyModel::new(vec![2.0, -3.0, 1.0], 2, 20, None, None, 0.0)
    }

    #[test]
    fn predict_quadratic() {
        let m = quad_model();
        assert!((m.predict(&[1.0]).unwrap() - 0.0).abs() < 1e-12);
        assert!((m.predict(&[2.0]).unwrap() - (-7.0)).abs() < 1e-12);
    }

    #[test]
    fn feature_count_check() {
        let m = quad_model();
        assert!(m.predict(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn bincode_roundtrip() {
        let m = quad_model();
        let bytes = MlModel::serialize(&m).unwrap();
        let back = <PolyModel as MlModel>::deserialize(&bytes).unwrap();
        assert_eq!(back.predict(&[1.5]).unwrap(), m.predict(&[1.5]).unwrap());
        assert_eq!(back.metadata.algorithm, Algorithm::PolynomialRegression);
    }
}
