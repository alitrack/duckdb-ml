use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::dbscan::{self, DbscanCluster};

/// DBSCAN clustering model
///
/// Stores each cluster's representative (mean) vector + size, plus `eps`.
/// `predict()` returns the nearest-cluster label (0..k-1) as f64, following
/// MADlib's simplified `dbscan_predict` semantics.
pub struct DbscanModel {
    pub metadata: ModelMetadata,
    clusters: Vec<DbscanCluster>,
    eps: f64,
}

impl DbscanModel {
    pub fn new(
        clusters: Vec<DbscanCluster>,
        num_features: usize,
        num_samples: usize,
        eps: f64,
        noise_count: usize,
    ) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::DBSCAN,
            num_features,
            num_samples,
            r_squared: None,
            mse: None,
            coefficients_count: clusters.len(),
            hyperparameters_json: serde_json::json!({
                "eps": eps,
                "clusters": clusters.len(),
                "noise": noise_count
            })
            .to_string(),
        };
        Self {
            metadata,
            clusters,
            eps,
        }
    }

    /// Reference to cluster representatives
    pub fn clusters(&self) -> &[DbscanCluster] {
        &self.clusters
    }

    /// Neighborhood radius used at fit time
    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl MlModel for DbscanModel {
    /// Returns the nearest-cluster label (0-indexed) for a feature vector
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        Ok(dbscan::nearest_cluster(features, &self.clusters))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::DBSCAN
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(dbscan::serialize(
            &self.clusters,
            self.metadata.num_features,
            self.eps,
        ))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let (clusters, nf, eps) = dbscan::deserialize(blob)
            .ok_or_else(|| ModelError::Serialization("Failed to decode DBSCAN model".into()))?;
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::DBSCAN,
                num_features: nf,
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: clusters.len(),
                hyperparameters_json: "{}".into(),
            },
            clusters,
            eps,
        })
    }
}
