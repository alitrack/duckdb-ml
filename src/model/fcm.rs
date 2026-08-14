//! FcmModel — MlModel wrapper for fuzzy c-Means.
//!
//! `predict()` returns the hard label (argmax membership = nearest centroid),
//! same convention as KMeansModel. The membership matrix itself is available
//! via train()'s FcmResult for soft-partition use cases.

use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::fcm;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct FcmModel {
    pub metadata: ModelMetadata,
    centroids: Vec<Vec<f64>>,
    fuzziness: f64,
}

impl FcmModel {
    pub fn new(
        centroids: Vec<Vec<f64>>,
        num_features: usize,
        num_samples: usize,
        k: usize,
        fuzziness: f64,
        iterations: usize,
    ) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::FuzzyCMeans,
            num_features,
            num_samples,
            r_squared: None,
            mse: None,
            coefficients_count: k,
            hyperparameters_json: serde_json::json!({
                "k": k,
                "fuzziness": fuzziness,
                "iterations": iterations,
            })
            .to_string(),
        };
        Self {
            metadata,
            centroids,
            fuzziness,
        }
    }

    pub fn k(&self) -> usize {
        self.centroids.len()
    }

    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }
}

impl MlModel for FcmModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        Ok(fcm::nearest_centroid(features, &self.centroids) as f64)
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::FuzzyCMeans
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

    fn model() -> FcmModel {
        FcmModel::new(
            vec![vec![0.0, 0.0], vec![6.0, 0.0]],
            2,
            100,
            2,
            2.0,
            12,
        )
    }

    #[test]
    fn predict_nearest() {
        let m = model();
        assert_eq!(m.predict(&[0.1, 0.1]).unwrap(), 0.0);
        assert_eq!(m.predict(&[5.9, -0.2]).unwrap(), 1.0);
    }

    #[test]
    fn feature_count_check() {
        let m = model();
        assert!(m.predict(&[1.0]).is_err());
    }

    #[test]
    fn bincode_roundtrip() {
        let m = model();
        let bytes = MlModel::serialize(&m).unwrap();
        let back = <FcmModel as MlModel>::deserialize(&bytes).unwrap();
        assert_eq!(back.predict(&[5.9, -0.2]).unwrap(), 1.0);
        assert_eq!(back.metadata.algorithm, Algorithm::FuzzyCMeans);
    }
}
