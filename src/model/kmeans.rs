use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::kmeans;

/// K-Means clustering model
///
/// `predict()` returns the cluster label (0..k-1) for a given feature vector.
pub struct KMeansModel {
    pub metadata: ModelMetadata,
    centroids: Vec<Vec<f64>>,
    inertia: f64,
}

impl KMeansModel {
    pub fn new(
        centroids: Vec<Vec<f64>>,
        num_features: usize,
        num_samples: usize,
        k: usize,
        inertia: f64,
        iterations: usize,
    ) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::KMeans,
            num_features,
            num_samples,
            r_squared: None,
            mse: Some(inertia),
            coefficients_count: k,
            hyperparameters_json: serde_json::json!({
                "k": k,
                "iterations": iterations,
                "inertia": inertia
            })
            .to_string(),
        };
        Self {
            metadata,
            centroids,
            inertia,
        }
    }

    /// Number of clusters
    pub fn k(&self) -> usize {
        self.centroids.len()
    }

    /// Reference to centroids
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// Sum of squared distances to nearest centroid
    pub fn inertia(&self) -> f64 {
        self.inertia
    }
}

impl MlModel for KMeansModel {
    /// Returns the cluster label (0-indexed) for a feature vector
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        let label = kmeans::nearest_centroid(features, &self.centroids);
        Ok(label as f64)
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::KMeans
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(kmeans::serialize_centroids(&self.centroids))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let (centroids, k, nf) = kmeans::deserialize_centroids(blob).ok_or_else(|| {
            ModelError::Serialization("Failed to decode k-means centroids".into())
        })?;
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::KMeans,
                num_features: nf,
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: k,
                hyperparameters_json: "{}".into(),
            },
            centroids,
            inertia: 0.0,
        })
    }
}
