use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::svm::{self, KERNEL_LINEAR};

/// Binary SVM classifier model (linfa-svm).
pub struct SvmModel {
    pub metadata: ModelMetadata,
    /// De-serialized blob; `predict` goes through linfa's own decision path.
    blob: svm::SvmBlob,
}

impl SvmModel {
    fn from_blob(blob: svm::SvmBlob) -> Self {
        let kernel_name = match blob.kernel {
            KERNEL_LINEAR => "linear",
            _ => "gaussian",
        };
        let n_support = blob.svm.nsupport();
        let rho = blob.svm.rho;
        let metadata = ModelMetadata {
            algorithm: Algorithm::SVM,
            num_features: blob.n_features,
            num_samples: 0,
            r_squared: None,
            mse: Some(rho),
            coefficients_count: n_support,
            hyperparameters_json: format!("kernel={kernel_name}, support_vectors={n_support}"),
        };
        Self { metadata, blob }
    }
}

impl MlModel for SvmModel {
    fn algorithm(&self) -> Algorithm {
        Algorithm::SVM
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        Ok(svm::predict_one(&self.blob.svm, features))
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        bincode::serde::encode_to_vec(&self.blob, bincode::config::standard())
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let b = svm::deserialize(blob).map_err(ModelError::Training)?;
        Ok(Self::from_blob(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_predict() {
        let x = vec![vec![-1.0], vec![-0.5], vec![1.0], vec![1.5]];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let t = svm::train(&x, &y, 1.0, KERNEL_LINEAR, 1.0).unwrap();
        let m = SvmModel::deserialize(&t.blob).unwrap();
        assert_eq!(m.predict(&[-0.9]).unwrap(), 0.0);
        assert_eq!(m.predict(&[1.2]).unwrap(), 1.0);
        // serialize → deserialize → predict again
        let blob = m.serialize().unwrap();
        let m2 = SvmModel::deserialize(&blob).unwrap();
        assert_eq!(m2.predict(&[2.0]).unwrap(), 1.0);
        assert_eq!(m2.predict(&[-2.0]).unwrap(), 0.0);
        assert_eq!(m2.metadata().algorithm, Algorithm::SVM);
        assert_eq!(m2.metadata().num_features, 1);
        // dimension check
        assert!(m2.predict(&[1.0, 2.0]).is_err());
    }
}
