//! AdaBoostModel — MlModel wrapper for the AdaBoost stump ensemble.
//!
//! Serialization: stumps/alphas/classes packed as flat f64 vectors in a
//! bincode envelope: [classes(2) | alphas(k) | stump triples (feature as f64,
//! threshold, left, right)].

use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::adaboost::{self, Stump};
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct AdaBoostModel {
    pub metadata: ModelMetadata,
    stumps: Vec<StumpData>,
    alphas: Vec<f64>,
    classes: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
struct StumpData {
    feature: usize,
    threshold: f64,
    left: f64,
    right: f64,
}

impl AdaBoostModel {
    pub fn new(
        result: &adaboost::AdaBoostResult,
        num_features: usize,
        num_samples: usize,
    ) -> Self {
        let stumps: Vec<StumpData> = result
            .stumps
            .iter()
            .map(|s| StumpData {
                feature: s.feature,
                threshold: s.threshold,
                left: s.left,
                right: s.right,
            })
            .collect();
        let metadata = ModelMetadata {
            algorithm: Algorithm::AdaBoost,
            num_features,
            num_samples,
            r_squared: None,
            mse: None,
            coefficients_count: result.stumps.len(),
            hyperparameters_json: serde_json::json!({
                "n_estimators": result.stumps.len(),
                "base": "decision_stump",
            })
            .to_string(),
        };
        Self {
            metadata,
            stumps,
            alphas: result.alphas.clone(),
            classes: result.classes.clone(),
        }
    }

    /// Rebuild a train-side AdaBoostResult view for predict().
    fn result(&self) -> adaboost::AdaBoostResult {
        adaboost::AdaBoostResult {
            stumps: self
                .stumps
                .iter()
                .map(|s| Stump {
                    feature: s.feature,
                    threshold: s.threshold,
                    left: s.left,
                    right: s.right,
                })
                .collect(),
            alphas: self.alphas.clone(),
            classes: self.classes.clone(),
        }
    }
}

impl MlModel for AdaBoostModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        Ok(adaboost::predict(&self.result(), features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::AdaBoost
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
    use crate::train::adaboost;

    fn separable() -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..30 {
            for j in 0..30 {
                let a = (i as f64 - 15.0) * 0.2;
                let b = (j as f64 - 15.0) * 0.2;
                x.push(vec![a, b]);
                y.push(if a + b > 0.0 { 1.0 } else { 0.0 });
            }
        }
        (x, y)
    }

    #[test]
    fn bincode_roundtrip() {
        let (x, y) = separable();
        let r = adaboost::train(&x, &y, 3);
        let m = AdaBoostModel::new(&r, 2, x.len());
        let bytes = MlModel::serialize(&m).unwrap();
        let back = <AdaBoostModel as MlModel>::deserialize(&bytes).unwrap();
        assert_eq!(back.predict(&[1.0, 0.5]).unwrap(), 1.0);
        assert_eq!(back.predict(&[-1.0, -0.5]).unwrap(), 0.0);
        assert_eq!(back.metadata.algorithm, Algorithm::AdaBoost);
    }

    #[test]
    fn feature_count_check() {
        let (x, y) = separable();
        let r = adaboost::train(&x, &y, 2);
        let m = AdaBoostModel::new(&r, 2, x.len());
        assert!(m.predict(&[1.0]).is_err());
    }
}
