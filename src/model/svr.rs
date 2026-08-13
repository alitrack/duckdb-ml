//! SVR model wrapper (ε-SVR via hand-written SMO, see train/svr.rs).
//! Blob = bincode of the full model (metadata + SvrModelData).

use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::svr::{self, SvrModelData};
use bincode::{Decode, Encode};

/// Support vector regression model.
#[derive(Debug, Clone, Encode, Decode)]
pub struct SvrModel {
    pub metadata: ModelMetadata,
    pub data: SvrModelData,
}

impl SvrModel {
    pub fn new(data: SvrModelData, num_samples: usize) -> Self {
        let n_features = data.n_features;
        let metadata = ModelMetadata {
            algorithm: Algorithm::SVR,
            num_features: n_features,
            num_samples,
            r_squared: data.r_squared,
            mse: data.mse,
            coefficients_count: data.support.len(),
            hyperparameters_json: serde_json::json!({
                "kernel": data.kernel.as_str(),
                "c": data.c,
                "epsilon": data.epsilon,
                "gamma": data.gamma,
                "degree": data.degree,
                "coef0": data.coef0
            })
            .to_string(),
        };
        Self { metadata, data }
    }

    pub fn from_data(data: SvrModelData, num_samples: usize) -> Self {
        Self::new(data, num_samples)
    }
}

impl MlModel for SvrModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.data.n_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.data.n_features,
                got: features.len(),
            });
        }
        Ok(svr::predict_one(&self.data, features))
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
        bincode::decode_from_slice(blob, bincode::config::standard())
            .map(|(m, _)| m)
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::svr::{train, Kernel};

    fn sample_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
        let y: Vec<f64> = (0..10).map(|i| 3.0 + 2.0 * i as f64 * 0.5).collect();
        (x, y)
    }

    #[test]
    fn svr_model_roundtrip() {
        let (x, y) = sample_data();
        let r = train(&x, &y, Kernel::Linear, 1.0, 1e-6, 0.0, 2, 0.0, 1e-4, 2000).unwrap();
        let data = svr::deserialize(r.model_blob.as_ref().unwrap()).unwrap();
        let m = SvrModel::from_data(data, r.num_samples);
        let bytes = m.serialize().unwrap();
        let m2 = SvrModel::deserialize(&bytes).unwrap();
        assert_eq!(m2.algorithm(), Algorithm::SVR);
        assert_eq!(m2.data.support.len(), m.data.support.len());
        let p = m2.predict(&[7.0]).unwrap();
        assert!((p - 17.0).abs() < 0.1, "pred={p}");
    }

    #[test]
    fn feature_count_mismatch() {
        let (x, y) = sample_data();
        let r = train(&x, &y, Kernel::Linear, 1.0, 1e-6, 0.0, 2, 0.0, 1e-4, 2000).unwrap();
        let data = svr::deserialize(r.model_blob.as_ref().unwrap()).unwrap();
        let m = SvrModel::from_data(data, r.num_samples);
        assert!(m.predict(&[1.0, 2.0]).is_err());
    }
}
