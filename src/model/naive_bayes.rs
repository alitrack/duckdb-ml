//! Gaussian Naive Bayes classifier
//!
//! Computes per-class priors and per-feature (mean, variance) for fast prediction.

#[derive(Debug, Clone)]
pub struct NaiveBayesModel {
    /// Prior probability for each class
    priors: Vec<f64>,
    /// Class labels corresponding to priors
    classes: Vec<f64>,
    /// Per-class per-feature mean: means[class_idx][feature_idx]
    means: Vec<Vec<f64>>,
    /// Per-class per-feature variance (with epsilon for stability)
    vars: Vec<Vec<f64>>,
}

impl NaiveBayesModel {
    /// Train Gaussian Naive Bayes. x: n_samples × n_features, y: targets
    pub fn train(x: &[Vec<f64>], y: &[f64]) -> Self {
        let n_samples = x.len();
        let n_features = x[0].len();

        // Collect unique classes
        let mut class_set = std::collections::BTreeMap::new();
        for &label in y {
            let key = (label * 1e9).round() as i64;
            class_set.entry(key).or_insert(label);
        }
        let classes: Vec<f64> = class_set.into_values().collect();
        let n_classes = classes.len();

        // Group samples by class
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (i, &label) in y.iter().enumerate() {
            for (c_idx, &c_label) in classes.iter().enumerate() {
                if (label - c_label).abs() < 1e-12 {
                    class_indices[c_idx].push(i);
                    break;
                }
            }
        }

        // Priors
        let priors: Vec<f64> = class_indices
            .iter()
            .map(|indices| indices.len() as f64 / n_samples as f64)
            .collect();

        // Per-class per-feature mean and variance
        let eps = 1e-9_f64;
        let mut means = vec![vec![0.0; n_features]; n_classes];
        let mut vars = vec![vec![eps; n_features]; n_classes];

        for (c_idx, indices) in class_indices.iter().enumerate() {
            let n_c = indices.len() as f64;
            if n_c < 1.0 {
                continue;
            }
            // Mean
            for &f_idx in indices {
                for j in 0..n_features {
                    means[c_idx][j] += x[f_idx][j] / n_c;
                }
            }
            // Variance
            for &f_idx in indices {
                for (j, (&x_v, &mean_v)) in x[f_idx].iter().zip(means[c_idx].iter()).enumerate() {
                    let diff = x_v - mean_v;
                    vars[c_idx][j] += diff * diff / n_c;
                }
            }
            // Add epsilon
            for v in &mut vars[c_idx] {
                *v += eps;
            }
        }

        Self {
            priors,
            classes,
            means,
            vars,
        }
    }

    /// Predict the most likely class
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut best_class = 0.0;
        let mut best_log_prob = f64::NEG_INFINITY;

        for (c_idx, class) in self.classes.iter().enumerate() {
            let log_prior = self.priors[c_idx].ln();
            let log_likelihood: f64 = features
                .iter()
                .zip(self.means[c_idx].iter())
                .zip(self.vars[c_idx].iter())
                .map(|((&f, &m), &v)| {
                    let diff = f - m;
                    -0.5 * (diff * diff / v + (2.0 * std::f64::consts::PI * v).ln())
                })
                .sum();
            let log_prob = log_prior + log_likelihood;
            if log_prob > best_log_prob {
                best_log_prob = log_prob;
                best_class = *class;
            }
        }
        best_class
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    pub fn n_features(&self) -> usize {
        if !self.means.is_empty() {
            self.means[0].len()
        } else {
            0
        }
    }

    // ——— Serialization ———

    pub fn to_bytes(&self) -> Vec<u8> {
        let n_classes = self.classes.len();
        let nf = self.n_features();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(n_classes as u32).to_le_bytes());
        buf.extend_from_slice(&(nf as u32).to_le_bytes());

        for &c in &self.classes {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        for &p in &self.priors {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        for row in &self.means {
            for &v in row {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for row in &self.vars {
            for &v in row {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        let n_classes = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let nf = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let expected = 8 + n_classes * 8 * 2 + n_classes * nf * 8 * 2;
        if data.len() < expected {
            return None;
        }

        let mut pos = 8;
        let mut classes = Vec::with_capacity(n_classes);
        let mut priors = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
            classes.push(f64::from_le_bytes(b));
            pos += 8;
        }
        for _ in 0..n_classes {
            let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
            priors.push(f64::from_le_bytes(b));
            pos += 8;
        }

        let mut means = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let mut row = Vec::with_capacity(nf);
            for _ in 0..nf {
                let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
                row.push(f64::from_le_bytes(b));
                pos += 8;
            }
            means.push(row);
        }

        let mut vars = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let mut row = Vec::with_capacity(nf);
            for _ in 0..nf {
                let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
                row.push(f64::from_le_bytes(b));
                pos += 8;
            }
            vars.push(row);
        }

        Some(Self {
            priors,
            classes,
            means,
            vars,
        })
    }
}

// ── MlModel wrapper ──

use crate::model::{Algorithm, MlModel, ModelError, ModelMetadata};

pub struct NbMlModel {
    pub metadata: ModelMetadata,
    pub inner: NaiveBayesModel,
}

impl MlModel for NbMlModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        Ok(self.inner.predict(features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::NaiveBayes
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(self.inner.to_bytes())
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError> {
        let inner = NaiveBayesModel::from_bytes(blob)
            .ok_or_else(|| ModelError::Serialization("Failed to decode NaiveBayes model".into()))?;
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::NaiveBayes,
                num_features: inner.n_features(),
                num_samples: 0, // not stored in blob
                r_squared: None,
                mse: None,
                coefficients_count: inner.n_classes() * inner.n_features(),
                hyperparameters_json: serde_json::json!({"n_classes": inner.n_classes(), "n_features": inner.n_features()}).to_string(),
            },
            inner,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_bayes() {
        // Two well-separated Gaussian blobs
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..50 {
            x.push(vec![1.0, 1.0]);
            y.push(0.0);
            x.push(vec![5.0, 5.0]);
            y.push(1.0);
        }

        let model = NaiveBayesModel::train(&x, &y);
        assert_eq!(model.n_classes(), 2);

        let p0 = model.predict(&[1.5, 1.5]);
        assert!((p0 - 0.0).abs() < 0.01, "p0={p0}");

        let p1 = model.predict(&[4.5, 4.5]);
        assert!((p1 - 1.0).abs() < 0.01, "p1={p1}");
    }

    #[test]
    fn test_nb_serialization() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y = vec![0.0, 1.0, 1.0];
        let model = NaiveBayesModel::train(&x, &y);
        let bytes = model.to_bytes();
        let recovered = NaiveBayesModel::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.n_classes(), 2);
        let pred = recovered.predict(&[4.0, 5.0]);
        assert!((pred - 1.0).abs() < 0.01, "pred={pred}");
    }

    #[test]
    fn test_nb_mlmodel_roundtrip() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let y = vec![0.0, 1.0, 1.0];
        let inner = NaiveBayesModel::train(&x, &y);
        let model = NbMlModel {
            metadata: ModelMetadata {
                algorithm: Algorithm::NaiveBayes,
                num_features: 2,
                num_samples: 3,
                r_squared: None,
                mse: None,
                coefficients_count: 4,
                hyperparameters_json: "{}".into(),
            },
            inner,
        };

        let blob = model.serialize().unwrap();
        let recovered = NbMlModel::deserialize(&blob).unwrap();
        assert_eq!(recovered.algorithm(), Algorithm::NaiveBayes);
        assert_eq!(recovered.inner.n_features(), 2);

        let orig = model.predict(&[4.0, 5.0]).unwrap();
        let recv = recovered.predict(&[4.0, 5.0]).unwrap();
        assert!((orig - recv).abs() < 0.01);
    }
}
