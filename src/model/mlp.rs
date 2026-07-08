//! Multi-Layer Perceptron (MLP) Regressor
//!
//! Single-hidden-layer neural network for regression.
//! Architecture: input → ReLU(hidden) → linear(output)
//! Trained via mini-batch SGD with momentum.

use super::types::{Algorithm, MlModel, ModelError, ModelMetadata};

/// MLP weights
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct MlpWeights {
    /// input → hidden weights: [hidden_size][input_size]
    pub w1: Vec<Vec<f64>>,
    /// hidden biases
    pub b1: Vec<f64>,
    /// hidden → output weights: [hidden_size]
    pub w2: Vec<f64>,
    /// output bias
    pub b2: f64,
    pub input_size: usize,
    pub hidden_size: usize,
}

/// MlModel wrapper
pub struct MlpModel {
    pub metadata: ModelMetadata,
    pub inner: MlpWeights,
}

impl MlpModel {
    pub fn new(weights: MlpWeights, r_squared: Option<f64>, mse: Option<f64>) -> Self {
        Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::MlpRegressor,
                num_features: weights.input_size,
                num_samples: 0,
                r_squared,
                mse,
                coefficients_count: weights.w1.len() * weights.w1[0].len()
                    + weights.b1.len()
                    + weights.w2.len()
                    + 1,
                hyperparameters_json: format!(
                    r#"{{"hidden_size":{},"iterations":0}}"#,
                    weights.hidden_size
                ),
            },
            inner: weights,
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(&self, features: &[f64]) -> f64 {
        let mut hidden = vec![0.0f64; self.inner.hidden_size];
        // Hidden layer: ReLU(W1 @ x + b1)
        for i in 0..self.inner.hidden_size {
            let mut z = self.inner.b1[i];
            for j in 0..features.len() {
                z += self.inner.w1[i][j] * features[j];
            }
            hidden[i] = if z > 0.0 { z } else { 0.0 };
        }
        // Output layer: W2 · hidden + b2
        let mut out = self.inner.b2;
        for i in 0..self.inner.hidden_size {
            out += self.inner.w2[i] * hidden[i];
        }
        out
    }
}

impl MlModel for MlpModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        Ok(self.forward(features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::MlpRegressor
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        bincode::encode_to_vec(&self.inner, bincode::config::standard())
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError> {
        let (weights, _): (MlpWeights, _) =
            bincode::decode_from_slice(blob, bincode::config::standard())
                .map_err(|e| ModelError::Serialization(e.to_string()))?;
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::MlpRegressor,
                num_features: weights.input_size,
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: weights.w1.len() * weights.w1[0].len()
                    + weights.b1.len()
                    + weights.w2.len()
                    + 1,
                hyperparameters_json: format!(
                    r#"{{"hidden_size":{},"iterations":0}}"#,
                    weights.hidden_size
                ),
            },
            inner: weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_serialization() {
        let w = MlpWeights {
            w1: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            b1: vec![0.5, -0.5],
            w2: vec![1.0, 2.0],
            b2: 0.1,
            input_size: 2,
            hidden_size: 2,
        };
        let model = MlpModel::new(w, None, None);
        let blob = model.serialize().unwrap();
        let restored = MlpModel::deserialize(&blob).unwrap();
        let pred = restored.predict(&[1.0, 1.0]).unwrap();
        assert!(pred.is_finite());
    }
}
