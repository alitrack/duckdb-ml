use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Linear regression model (OLS or Ridge)
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct LinearModel {
    pub metadata: ModelMetadata,
    /// Coefficients including intercept as last element
    coefficients: Vec<f64>,
    lambda: f64,
}

impl LinearModel {
    pub fn new(
        coefficients: Vec<f64>,
        num_samples: usize,
        r_squared: Option<f64>,
        mse: Option<f64>,
        lambda: f64,
    ) -> Self {
        let algo = if lambda > 0.0 {
            Algorithm::RidgeRegression
        } else {
            Algorithm::LinearRegression
        };
        let metadata = ModelMetadata {
            algorithm: algo,
            num_features: coefficients.len() - 1, // minus intercept
            num_samples,
            r_squared,
            mse,
            coefficients_count: coefficients.len(),
            hyperparameters_json: serde_json::json!({"lambda": lambda}).to_string(),
        };
        Self {
            metadata,
            coefficients,
            lambda,
        }
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

impl MlModel for LinearModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() + 1 != self.coefficients.len() {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.coefficients.len() - 1,
                got: features.len(),
            });
        }
        let mut result = self.coefficients.last().copied().unwrap_or(0.0);
        for (i, &x) in features.iter().enumerate() {
            result += self.coefficients[i] * x;
        }
        Ok(result)
    }

    fn algorithm(&self) -> Algorithm {
        self.metadata.algorithm
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
