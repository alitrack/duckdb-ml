//! K-Nearest Neighbors — lazy learning for regression and classification.
//!
//! Zero training. Stores all data; predicts by averaging k nearest neighbor targets.

/// KNN model: stores the entire training dataset
#[derive(Debug, Clone)]
pub struct KnnModel {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: usize,
    task: KnnTask,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KnnTask {
    Regression,
    Classification,
}

impl KnnModel {
    pub fn new(x_train: Vec<Vec<f64>>, y_train: Vec<f64>, k: usize, task: KnnTask) -> Self {
        assert!(k > 0 && k <= x_train.len());
        Self {
            x_train,
            y_train,
            k,
            task,
        }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let n = self.x_train.len();
        let k = self.k.min(n);

        // Compute all distances and keep top-k with their target values
        let mut heap: Vec<(f64, f64)> = Vec::with_capacity(n);

        for (i, row) in self.x_train.iter().enumerate() {
            let dist: f64 = row
                .iter()
                .zip(features.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            heap.push((dist, self.y_train[i]));
        }

        // Partial sort to get k smallest distances
        heap.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());

        match self.task {
            KnnTask::Regression => {
                let sum: f64 = heap[..k].iter().map(|(_, y)| *y).sum();
                sum / k as f64
            }
            KnnTask::Classification => {
                // Majority vote among k nearest
                let mut votes = std::collections::HashMap::new();
                for &(_, y) in &heap[..k] {
                    let bucket = (y * 1000.0).round() as i64; // discretize for tolerance
                    *votes.entry(bucket).or_insert(0u32) += 1;
                }
                votes
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(bucket, _)| bucket as f64 / 1000.0)
                    .unwrap_or(0.0)
            }
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n_samples(&self) -> usize {
        self.x_train.len()
    }

    pub fn n_features(&self) -> usize {
        if !self.x_train.is_empty() {
            self.x_train[0].len()
        } else {
            0
        }
    }

    // ——— Serialization ———

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Header: k(u32) + task(u8) + samples(u32) + features(u32)
        buf.extend_from_slice(&(self.k as u32).to_le_bytes());
        buf.push(self.task as u8);
        buf.extend_from_slice(&(self.x_train.len() as u32).to_le_bytes());
        let n_feat = if self.x_train.is_empty() {
            0
        } else {
            self.x_train[0].len()
        };
        buf.extend_from_slice(&(n_feat as u32).to_le_bytes());
        // Data
        for row in &self.x_train {
            for &v in row {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &y in &self.y_train {
            buf.extend_from_slice(&y.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 13 {
            return None;
        }
        let k = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let task = match data[4] {
            0 => KnnTask::Regression,
            1 => KnnTask::Classification,
            _ => return None,
        };
        let n_samples = u32::from_le_bytes(data[5..9].try_into().ok()?) as usize;
        let n_features = u32::from_le_bytes(data[9..13].try_into().ok()?) as usize;

        let expected = 13 + n_samples * n_features * 8 + n_samples * 8;
        if data.len() < expected {
            return None;
        }

        let mut pos = 13;
        let mut x_train = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut row = Vec::with_capacity(n_features);
            for _ in 0..n_features {
                let bytes: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
                row.push(f64::from_le_bytes(bytes));
                pos += 8;
            }
            x_train.push(row);
        }

        let mut y_train = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let bytes: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
            y_train.push(f64::from_le_bytes(bytes));
            pos += 8;
        }

        Some(Self {
            x_train,
            y_train,
            k,
            task,
        })
    }
}

// ── MlModel wrapper ──

use crate::model::{Algorithm, MlModel, ModelError, ModelMetadata};

pub struct KnnMlModel {
    pub metadata: ModelMetadata,
    pub inner: KnnModel,
}

impl MlModel for KnnMlModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        Ok(self.inner.predict(features))
    }

    fn algorithm(&self) -> Algorithm {
        self.metadata.algorithm
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(self.inner.to_bytes())
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError> {
        let inner = KnnModel::from_bytes(blob)
            .ok_or_else(|| ModelError::Serialization("Failed to decode KNN model".into()))?;
        let task_str = match inner.task {
            KnnTask::Regression => "regression",
            KnnTask::Classification => "classification",
        };
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: match inner.task {
                    KnnTask::Regression => Algorithm::KNNRegressor,
                    KnnTask::Classification => Algorithm::KNNClassifier,
                },
                num_features: inner.n_features(),
                num_samples: inner.n_samples(),
                r_squared: None,
                mse: None,
                coefficients_count: inner.n_samples(),
                hyperparameters_json: serde_json::json!({"k": inner.k(), "task": task_str})
                    .to_string(),
            },
            inner,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_regression() {
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
            vec![6.0, 6.0],
        ];
        let y = vec![0.0, 2.0, 4.0, 10.0, 12.0];

        let model = KnnModel::new(x, y, 3, KnnTask::Regression);

        // [3.0, 3.0] nearest: [2,2](4.0), [5,5](10.0), [1,1](2.0) → avg=5.33
        let pred = model.predict(&[3.0, 3.0]);
        assert!((pred - 5.333).abs() < 0.1, "pred={pred}");

        // [5.5, 5.5] nearest: [5,5](10.0), [6,6](12.0), [2,2](4.0) → avg=8.667
        let pred2 = model.predict(&[5.5, 5.5]);
        assert!((pred2 - 8.667).abs() < 0.1, "pred={pred2}");
    }

    #[test]
    fn test_knn_classification() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
            vec![6.0, 6.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = KnnModel::new(x, y, 3, KnnTask::Classification);

        // [2.0, 2.0] → closer to class 0 cluster
        let pred = model.predict(&[2.0, 2.0]);
        assert!((pred - 0.0).abs() < 0.01, "pred={pred}");

        // [4.0, 5.0] → closer to class 1 cluster
        let pred2 = model.predict(&[4.0, 5.0]);
        assert!((pred2 - 1.0).abs() < 0.01, "pred={pred2}");
    }

    #[test]
    fn test_knn_serialization() {
        // Non-equidistant to avoid tie in k=1 nearest neighbor selection
        let x = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let y = vec![10.0, 20.0];
        let model = KnnModel::new(x, y, 1, KnnTask::Regression);

        let bytes = model.to_bytes();
        let recovered = KnnModel::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.k, 1);
        assert_eq!(recovered.x_train, model.x_train);
        assert_eq!(recovered.y_train, model.y_train);

        let pred = recovered.predict(&[2.0, 3.0]);
        // dist [0,0]: sqrt(4+9)=3.606, dist [3,4]: sqrt(1+1)=1.414 → nearest is 20.0
        assert!((pred - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_knn_mlmodel_roundtrip() {
        let x = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let y = vec![10.0, 20.0];
        let inner = KnnModel::new(x, y, 1, KnnTask::Regression);
        let model = KnnMlModel {
            metadata: ModelMetadata {
                algorithm: Algorithm::KNNRegressor,
                num_features: 2,
                num_samples: 2,
                r_squared: None,
                mse: None,
                coefficients_count: 2,
                hyperparameters_json: "{}".into(),
            },
            inner: inner.clone(),
        };

        // Serialize → deserialize roundtrip
        let blob = model.serialize().unwrap();
        let recovered = KnnMlModel::deserialize(&blob).unwrap();
        assert_eq!(recovered.algorithm(), Algorithm::KNNRegressor);
        assert_eq!(recovered.inner.n_features(), 2);

        // Predict should match
        let orig = model.predict(&[2.0, 3.0]).unwrap();
        let recv = recovered.predict(&[2.0, 3.0]).unwrap();
        assert!((orig - recv).abs() < 0.01);
    }
}
