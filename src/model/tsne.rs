//! TsneModel — MlModel wrapper for t-SNE embeddings.
//!
//! Stores the 2D embedding + training rows; predict(x) = embedding of the
//! nearest training row (nearest-neighbor out-of-sample approximation) →
//! returns the x-coordinate of that embedding. Bincode serialization.

use super::types::ModelMetadata;
use crate::model::{Algorithm, MlModel, ModelError};
use crate::train::tsne::TsneResult;

#[derive(bincode::Encode, bincode::Decode)]
pub struct TsneModel {
    pub embedding: Vec<[f64; 2]>,
    pub x: Vec<Vec<f64>>,
    metadata: ModelMetadata,
}

impl TsneModel {
    pub fn new(result: &TsneResult, n_features: usize, n_samples: usize) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::TSNE,
            num_features: n_features,
            num_samples: n_samples,
            r_squared: None,
            mse: None,
            coefficients_count: 2, // 2D embedding
            hyperparameters_json: serde_json::json!({
                "components": 2,
                "kl_divergence": result.kl,
            })
            .to_string(),
        };
        Self {
            embedding: result.embedding.clone(),
            x: result.x.clone(),
            metadata,
        }
    }

    fn nearest_row(&self, f: &[f64]) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, row) in self.x.iter().enumerate() {
            let d: f64 = row
                .iter()
                .zip(f.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
}

impl MlModel for TsneModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        let i = self.nearest_row(features);
        Ok(self.embedding[i][0])
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::TSNE
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

    #[test]
    fn nearest_row_mapping() {
        let r = TsneResult {
            embedding: vec![[0.5, 0.0], [9.5, 0.0]],
            x: vec![vec![1.0, 1.0], vec![10.0, 10.0]],
            kl: 1.0,
        };
        let m = TsneModel::new(&r, 2, 2);
        assert_eq!(m.predict(&[1.1, 1.1]).unwrap(), 0.5);
        assert_eq!(m.predict(&[9.9, 9.9]).unwrap(), 9.5);
    }

    #[test]
    fn bincode_roundtrip() {
        let r = TsneResult {
            embedding: vec![[1.0, 2.0], [3.0, 4.0]],
            x: vec![vec![0.0], vec![1.0]],
            kl: 0.5,
        };
        let m = TsneModel::new(&r, 1, 2);
        let bytes = MlModel::serialize(&m).unwrap();
        let m2 = TsneModel::deserialize(&bytes).unwrap();
        assert_eq!(m2.predict(&[0.01]).unwrap(), 1.0);
        assert_eq!(m2.algorithm(), Algorithm::TSNE);
    }
}
