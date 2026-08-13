use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::cox::{self, CoxModel};

/// Cox proportional hazards model — `predict` returns relative risk exp(w·x).
pub struct CoxMlModel {
    pub metadata: ModelMetadata,
    model: CoxModel,
}

impl CoxMlModel {
    pub fn new(model: CoxModel) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::CoxProportionalHazards,
            num_features: model.n_features,
            num_samples: 0,
            r_squared: None,
            mse: None,
            coefficients_count: model.weights.len(),
            hyperparameters_json: "partial_likelihood, breslow_ties".into(),
        };
        Self { metadata, model }
    }
}

impl MlModel for CoxMlModel {
    fn algorithm(&self) -> Algorithm {
        Algorithm::CoxProportionalHazards
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
        Ok(cox::predict_one(&self.model, features))
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(cox::serialize(&self.model))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let m = cox::deserialize(blob).map_err(ModelError::Training)?;
        Ok(Self::new(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let x = vec![vec![1.0], vec![0.0], vec![1.0], vec![0.0]];
        let time = vec![2.0, 4.0, 5.0, 9.0];
        let event = vec![1.0, 1.0, 1.0, 1.0];
        let m = cox::train(&x, &time, &event, 0.05, 800).unwrap();
        let model = CoxMlModel::new(m);
        let blob = model.serialize().unwrap();
        let model2 = CoxMlModel::deserialize(&blob).unwrap();
        assert_eq!(
            model2.predict(&[1.0]).unwrap(),
            model.predict(&[1.0]).unwrap()
        );
        assert_eq!(
            model2.metadata().algorithm,
            Algorithm::CoxProportionalHazards
        );
        assert!(model2.predict(&[1.0, 2.0]).is_err());
    }
}
