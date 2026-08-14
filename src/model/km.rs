//! KmMlModel — MlModel wrapper for Kaplan-Meier survival curves.
//!
//! No features: `predict(&[])` returns the median survival time (first t with
//! S(t) ≤ 0.5), or the last survival time if the curve never crosses 0.5.

use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::km::{self, KmResult};
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct KmMlModel {
    pub metadata: ModelMetadata,
    times: Vec<f64>,
    survival: Vec<f64>,
    events: usize,
}

impl KmMlModel {
    pub fn new(result: &KmResult) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::KaplanMeier,
            num_features: 0,
            num_samples: result.n,
            r_squared: None,
            mse: None,
            coefficients_count: result.events,
            hyperparameters_json: serde_json::json!({
                "events": result.events,
                "censored": result.n - result.events,
                "curve_points": result.times.len(),
            })
            .to_string(),
        };
        Self {
            metadata,
            times: result.times.clone(),
            survival: result.survival.clone(),
            events: result.events,
        }
    }

    /// Survival probability at a given time (last value at/before t).
    pub fn survival_at(&self, t: f64) -> f64 {
        let mut s = 1.0f64;
        for (i, &ti) in self.times.iter().enumerate() {
            if ti <= t {
                s = self.survival[i];
            } else {
                break;
            }
        }
        s
    }

    pub fn curve(&self) -> (&[f64], &[f64]) {
        (&self.times, &self.survival)
    }
}

impl MlModel for KmMlModel {
    // Feature-less model: predict() ignores the (dummy) feature vector and
    // returns the median survival time. Keeping a permissive signature lets
    // callers pass any single column (DuckDB tables need >= 1 column).
    fn predict(&self, _features: &[f64]) -> Result<f64, ModelError> {
        Ok(km::median_survival(&KmResult {
            times: self.times.clone(),
            survival: self.survival.clone(),
            n: self.metadata.num_samples,
            events: self.events,
        })
        .unwrap_or_else(|| {
            // curve never crosses 0.5: report the last event time (or 0)
            self.times.last().copied().unwrap_or(0.0)
        }))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::KaplanMeier
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
    fn median_prediction() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1.0, 1.0, 1.0];
        let m = KmMlModel::new(&km::train(&time, &event));
        assert_eq!(m.predict(&[]).unwrap(), 2.0);
        // survival_at: S(0.5)=1 (before first event), S(1)=2/3, S(2.5)=1/3
        assert!((m.survival_at(0.5) - 1.0).abs() < 1e-12);
        assert!((m.survival_at(1.0) - 2.0 / 3.0).abs() < 1e-12);
        assert!((m.survival_at(2.5) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn dummy_features_ignored() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1.0, 1.0, 1.0];
        let m = KmMlModel::new(&km::train(&time, &event));
        // permissive: any dummy column is accepted (DuckDB tables need ≥ 1 col)
        assert_eq!(m.predict(&[42.0, -7.0]).unwrap(), 2.0);
        assert_eq!(m.predict(&[]).unwrap(), 2.0);
    }

    #[test]
    fn bincode_roundtrip() {
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1.0, 0.0, 1.0, 0.0];
        let m = KmMlModel::new(&km::train(&time, &event));
        let bytes = MlModel::serialize(&m).unwrap();
        let back = <KmMlModel as MlModel>::deserialize(&bytes).unwrap();
        assert_eq!(back.predict(&[]).unwrap(), m.predict(&[]).unwrap());
        assert_eq!(back.metadata.algorithm, Algorithm::KaplanMeier);
        assert_eq!(back.events, 2);
    }
}
