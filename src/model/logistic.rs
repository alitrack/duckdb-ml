use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Logistic regression model (binary classification)
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct LogisticModel {
    pub metadata: ModelMetadata,
    coefficients: Vec<f64>, // includes intercept as last element
}

impl LogisticModel {
    pub fn new(coefficients: Vec<f64>, num_samples: usize, accuracy: Option<f64>) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::LogisticRegression,
            num_features: coefficients.len() - 1,
            num_samples,
            r_squared: None,
            mse: None,
            coefficients_count: coefficients.len(),
            hyperparameters_json: serde_json::json!({"accuracy": accuracy}).to_string(),
        };
        Self {
            metadata,
            coefficients,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub fn predict_proba(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() + 1 != self.coefficients.len() {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.coefficients.len() - 1,
                got: features.len(),
            });
        }
        let mut z = self.coefficients.last().copied().unwrap_or(0.0);
        for (i, &x) in features.iter().enumerate() {
            z += self.coefficients[i] * x;
        }
        Ok(Self::sigmoid(z))
    }
}

impl MlModel for LogisticModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        let proba = self.predict_proba(features)?;
        Ok(if proba >= 0.5 { 1.0 } else { 0.0 })
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::LogisticRegression
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        bincode::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let (model, _): (Self, _) = bincode::decode_from_slice(blob, bincode::config::standard())
            .map_err(|e| ModelError::Serialization(e.to_string()))?;
        Ok(model)
    }
}
