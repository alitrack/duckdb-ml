use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::ordinal::{self, OrdinalModel};

/// Ordinal logistic regression (cumulative logit) model.
pub struct OrdinalMlModel {
    pub metadata: ModelMetadata,
    model: OrdinalModel,
}

impl OrdinalMlModel {
    pub fn new(model: OrdinalModel) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::OrdinalLogisticRegression,
            num_features: model.n_features,
            num_samples: 0,
            r_squared: None,
            mse: None,
            coefficients_count: model.weights.len() + model.thresholds.len(),
            hyperparameters_json: format!("classes={:?}", model.classes),
        };
        Self { metadata, model }
    }
}

impl MlModel for OrdinalMlModel {
    fn algorithm(&self) -> Algorithm {
        Algorithm::OrdinalLogisticRegression
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
        Ok(ordinal::predict_one(&self.model, features))
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(ordinal::serialize(&self.model))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let m = ordinal::deserialize(blob).map_err(ModelError::Training)?;
        Ok(Self::new(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let x = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let m = ordinal::train(&x, &y, 0.1, 1000).unwrap();
        let model = OrdinalMlModel::new(m);
        assert_eq!(model.predict(&[0.0]).unwrap(), 0.0);
        assert_eq!(model.predict(&[5.0]).unwrap(), 2.0);
        let blob = model.serialize().unwrap();
        let model2 = OrdinalMlModel::deserialize(&blob).unwrap();
        assert_eq!(model2.predict(&[3.0]).unwrap(), 1.0);
        assert_eq!(
            model2.metadata().algorithm,
            Algorithm::OrdinalLogisticRegression
        );
        assert!(model2.predict(&[1.0, 2.0]).is_err());
    }
}
