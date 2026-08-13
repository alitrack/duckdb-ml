use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::multilogistic::{self, MultinomialModel};

/// Multinomial logistic regression (softmax) model — multi-class labels.
pub struct MultilogisticModel {
    pub metadata: ModelMetadata,
    model: MultinomialModel,
}

impl MultilogisticModel {
    pub fn new(model: MultinomialModel) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::MultinomialLogisticRegression,
            num_features: model.n_features,
            num_samples: 0,
            r_squared: None,
            mse: None,
            coefficients_count: model.weights.len() * (model.n_features + 1),
            hyperparameters_json: format!("classes={:?}", model.classes),
        };
        Self { metadata, model }
    }
}

impl MlModel for MultilogisticModel {
    fn algorithm(&self) -> Algorithm {
        Algorithm::MultinomialLogisticRegression
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
        Ok(multilogistic::predict_one(&self.model, features))
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(multilogistic::serialize(&self.model))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let m = multilogistic::deserialize(blob).map_err(ModelError::Training)?;
        Ok(Self::new(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_predict() {
        let x = vec![vec![-3.0], vec![-2.5], vec![2.5], vec![3.0]];
        let y = vec![5.0, 5.0, 9.0, 9.0];
        let m = multilogistic::train(&x, &y, 0.1, 1000).unwrap();
        let mut model = MultilogisticModel::new(m);
        assert_eq!(model.predict(&[-2.0]).unwrap(), 5.0);
        assert_eq!(model.predict(&[2.0]).unwrap(), 9.0);
        // serialize → deserialize → predict
        let blob = model.serialize().unwrap();
        let model2 = MultilogisticModel::deserialize(&blob).unwrap();
        assert_eq!(model2.predict(&[3.0]).unwrap(), 9.0);
        assert_eq!(
            model2.metadata().algorithm,
            Algorithm::MultinomialLogisticRegression
        );
        assert_eq!(model2.metadata().num_features, 1);
        assert!(model2.predict(&[1.0, 2.0]).is_err());
    }
}
