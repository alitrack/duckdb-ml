//! AgglomerativeModel — MlModel wrapper for hierarchical clustering.
//!
//! Stores centroids + label order; predict = nearest centroid (same contract
//! as kmeans/fcm hard labels). Bincode serialization.

use super::types::ModelMetadata;
use crate::model::{Algorithm, MlModel, ModelError};
use crate::train::agglomerative::{AggResult, Linkage};

#[derive(bincode::Encode, bincode::Decode)]
pub struct AgglomerativeModel {
    pub centers: Vec<Vec<f64>>,
    pub linkage: String,
    metadata: ModelMetadata,
}
impl AgglomerativeModel {
    pub fn new(result: &AggResult, n_features: usize, n_samples: usize, linkage: Linkage) -> Self {
        let linkage_name = match linkage {
            Linkage::Single => "single",
            Linkage::Complete => "complete",
            Linkage::Average => "average",
        };
        let k = result.centers.len();
        let metadata = ModelMetadata {
            algorithm: Algorithm::Agglomerative,
            num_features: n_features,
            num_samples: n_samples,
            r_squared: None,
            mse: None,
            coefficients_count: k,
            hyperparameters_json: serde_json::json!({
                "linkage": linkage_name,
                "n_clusters": k,
            })
            .to_string(),
        };
        Self {
            centers: result.centers.clone(),
            linkage: linkage_name.into(),
            metadata,
        }
    }

    fn nearest_center(&self, f: &[f64]) -> usize {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, c) in self.centers.iter().enumerate() {
            let d: f64 = c
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

impl MlModel for AgglomerativeModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        Ok(self.nearest_center(features) as f64)
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::Agglomerative
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
    fn nearest_center_contract() {
        let r = AggResult {
            labels: vec![0, 1, 0],
            centers: vec![vec![0.0, 0.0], vec![10.0, 10.0]],
        };
        let m = AgglomerativeModel::new(&r, 2, 3, Linkage::Complete);
        assert_eq!(m.predict(&[1.0, 1.0]).unwrap(), 0.0);
        assert_eq!(m.predict(&[9.0, 9.0]).unwrap(), 1.0);
    }

    #[test]
    fn bincode_roundtrip() {
        let r = AggResult {
            labels: vec![0, 1],
            centers: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        let m = AgglomerativeModel::new(&r, 2, 2, Linkage::Single);
        let bytes = MlModel::serialize(&m).unwrap();
        let m2 = AgglomerativeModel::deserialize(&bytes).unwrap();
        assert_eq!(m2.predict(&[1.1, 2.1]).unwrap(), 0.0);
        assert_eq!(m2.algorithm(), Algorithm::Agglomerative);
    }
}
