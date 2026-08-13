use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::arima::{self, ArimaModel};

/// ARIMA(p,d,q) model — `predict([h])` returns the h-step-ahead forecast.
pub struct ArimaMlModel {
    pub metadata: ModelMetadata,
    model: ArimaModel,
}

impl ArimaMlModel {
    pub fn new(model: ArimaModel) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::Arima,
            num_features: 1, // single series; predict input = [h]
            num_samples: 0,
            r_squared: None,
            mse: None,
            coefficients_count: model.ar.len() + model.ma.len(),
            hyperparameters_json: format!(
                "p={},d={},q={},intercept={:.6}",
                model.p, model.d, model.q, model.intercept
            ),
        };
        Self { metadata, model }
    }
}

impl MlModel for ArimaMlModel {
    fn algorithm(&self) -> Algorithm {
        Algorithm::Arima
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != 1 {
            return Err(ModelError::FeatureCountMismatch {
                expected: 1,
                got: features.len(),
            });
        }
        let h = features[0];
        if !(1.0..=100_000.0).contains(&h) {
            return Err(ModelError::Training(format!(
                "arima: forecast horizon must be in [1, 100000], got {h}"
            )));
        }
        Ok(arima::forecast(&self.model, h as usize))
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(arima::serialize(&self.model))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let m = arima::deserialize(blob).map_err(ModelError::Training)?;
        Ok(Self::new(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let y: Vec<f64> = (0..20).map(|t| 5.0 + 1.5 * t as f64).collect();
        let m = arima::train(&y, 1, 1, 0, 0.05, 500).unwrap();
        let model = ArimaMlModel::new(m);
        let f1 = model.predict(&[1.0]).unwrap();
        assert!((f1 - 35.0).abs() < 0.1, "f1={f1}");
        let blob = model.serialize().unwrap();
        let model2 = ArimaMlModel::deserialize(&blob).unwrap();
        assert_eq!(model2.predict(&[1.0]).unwrap(), f1);
        assert_eq!(model2.metadata().algorithm, Algorithm::Arima);
        assert!(model2.predict(&[1.0, 2.0]).is_err());
        assert!(model2.predict(&[0.0]).is_err()); // h < 1
    }
}
