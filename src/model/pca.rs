//! PCA (Principal Component Analysis) — dimensionality reduction via covariance eigendecomposition.
//!
//! Uses power iteration to extract top-k principal components.

#[derive(Debug, Clone)]
pub struct PcaModel {
    /// Mean of training data (for centering)
    mean: Vec<f64>,
    /// Top-k principal components: components[comp_idx][feature_idx]
    components: Vec<Vec<f64>>,
    /// Explained variance ratio per component
    explained_variance_ratio: Vec<f64>,
}

impl PcaModel {
    /// Fit PCA to data, extracting top-k components.
    pub fn fit(x: &[Vec<f64>], k: usize) -> Self {
        let n_samples = x.len();
        let n_features = x[0].len();

        // 1. Center the data
        let mut mean = vec![0.0_f64; n_features];
        for row in x {
            for j in 0..n_features {
                mean[j] += row[j];
            }
        }
        for v in mean.iter_mut() {
            *v /= n_samples as f64;
        }

        // 2. Build covariance matrix (n_features × n_features, symmetric)
        let mut cov = vec![0.0_f64; n_features * n_features];
        for row in x {
            for j in 0..n_features {
                let dj = row[j] - mean[j];
                for l in 0..n_features {
                    cov[j * n_features + l] += dj * (row[l] - mean[l]);
                }
            }
        }
        for v in cov.iter_mut() {
            *v /= n_samples as f64;
        }

        // 3. Power iteration for top-k eigenvectors
        let total_variance: f64 = (0..n_features).map(|i| cov[i * n_features + i]).sum();

        let mut components = Vec::with_capacity(k);
        let mut explained = Vec::with_capacity(k);

        for comp in 0..k {
            let v = power_iteration(&cov, n_features, &components);
            let ev = rayleigh_quotient(&cov, n_features, &v);
            components.push(v.clone());
            let ratio = if total_variance > 0.0 {
                ev / total_variance
            } else {
                0.0
            };
            explained.push(ratio);

            // Deflate: subtract component from covariance
            for j in 0..n_features {
                for l in 0..n_features {
                    cov[j * n_features + l] -= ev * v[j] * v[l];
                }
            }

            if comp > 0 && explained[comp] < 0.001 * explained[0] {
                break;
            }
        }

        Self {
            mean,
            components,
            explained_variance_ratio: explained,
        }
    }

    /// Transform data to k-dimensional PCA space.
    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        self.components
            .iter()
            .map(|comp| {
                comp.iter()
                    .zip(features.iter())
                    .zip(self.mean.iter())
                    .map(|((&c, &x), &m)| c * (x - m))
                    .sum()
            })
            .collect()
    }

    /// Number of extracted components
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Original feature count
    pub fn n_features(&self) -> usize {
        self.mean.len()
    }

    /// Explained variance ratios
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
    }

    // For MlModel compatibility: PCA doesn't "predict" a single value.
    // Use transform() instead. predict() returns the first PC score.
    pub fn score(&self, features: &[f64]) -> f64 {
        self.transform(features).first().copied().unwrap_or(0.0)
    }

    // ——— Serialization ———

    pub fn to_bytes(&self) -> Vec<u8> {
        let nf = self.n_features();
        let nc = self.n_components();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(nf as u32).to_le_bytes());
        buf.extend_from_slice(&(nc as u32).to_le_bytes());

        for &v in &self.mean {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for row in &self.components {
            for &v in row {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for &v in &self.explained_variance_ratio {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        let nf = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let nc = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let expected = 8 + nf * 8 + nc * nf * 8 + nc * 8;
        if data.len() < expected {
            return None;
        }

        let mut pos = 8;
        let mut mean = Vec::with_capacity(nf);
        for _ in 0..nf {
            let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
            mean.push(f64::from_le_bytes(b));
            pos += 8;
        }

        let mut components = Vec::with_capacity(nc);
        for _ in 0..nc {
            let mut row = Vec::with_capacity(nf);
            for _ in 0..nf {
                let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
                row.push(f64::from_le_bytes(b));
                pos += 8;
            }
            components.push(row);
        }

        let mut evr = Vec::with_capacity(nc);
        for _ in 0..nc {
            let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
            evr.push(f64::from_le_bytes(b));
            pos += 8;
        }

        Some(Self {
            mean,
            components,
            explained_variance_ratio: evr,
        })
    }
}

/// Power iteration: find dominant eigenvector of a symmetric matrix
fn power_iteration(a: &[f64], n: usize, existing: &[Vec<f64>]) -> Vec<f64> {
    let mut v = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..50 {
        // Orthogonalize against existing eigenvectors
        for ev in existing {
            let dot: f64 = v.iter().zip(ev.iter()).map(|(&a, &b)| a * b).sum();
            for (vi, &ei) in v.iter_mut().zip(ev.iter()) {
                *vi -= dot * ei;
            }
        }

        // Matrix-vector multiply: w = A @ v
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[i * n + j] * v[j];
            }
        }

        // Normalize
        let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            break;
        }
        for (vi, &wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }

    v
}

/// Rayleigh quotient: v^T A v / v^T v (≈ eigenvalue)
fn rayleigh_quotient(a: &[f64], n: usize, v: &[f64]) -> f64 {
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += a[i * n + j] * v[j];
        }
    }
    let numer: f64 = v.iter().zip(av.iter()).map(|(&a, &b)| a * b).sum();
    let denom: f64 = v.iter().map(|&x| x * x).sum();
    if denom > 1e-14 { numer / denom } else { 0.0 }
}

// ── MlModel wrapper ──

use crate::model::{Algorithm, MlModel, ModelError, ModelMetadata};

pub struct PcaMlModel {
    pub metadata: ModelMetadata,
    pub inner: PcaModel,
}

impl MlModel for PcaMlModel {
    /// Returns the first principal component score (use transform() for full projection)
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        Ok(self.inner.score(features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::PCA
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(self.inner.to_bytes())
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError> {
        let inner = PcaModel::from_bytes(blob)
            .ok_or_else(|| ModelError::Serialization("Failed to decode PCA model".into()))?;
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::PCA,
                num_features: inner.n_features(),
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: inner.n_components() * inner.n_features(),
                hyperparameters_json: serde_json::json!({
                    "n_components": inner.n_components(),
                    "explained_variance_ratio": inner.explained_variance_ratio()
                })
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
    fn test_pca_simple() {
        // 2D data strongly correlated
        let x: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
        ];
        let pca = PcaModel::fit(&x, 2);
        assert_eq!(pca.n_components(), 2);
        assert!(pca.explained_variance_ratio[0] > 0.9);

        // Transform should give one strong component and one weak one
        let t = pca.transform(&[3.0, 6.0]);
        assert!(t.len() >= 1);
        assert!(
            t[1].abs() < 0.01,
            "second component should be near zero: {}",
            t[1]
        );
    }

    #[test]
    fn test_pca_serialization() {
        let x = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let pca = PcaModel::fit(&x, 1);
        let bytes = pca.to_bytes();
        let recovered = PcaModel::from_bytes(&bytes).unwrap();
        assert_eq!(recovered.n_components(), 1);
        assert_eq!(recovered.n_features(), 2);

        let t1 = pca.score(&[4.0, 5.0]);
        let t2 = recovered.score(&[4.0, 5.0]);
        assert!((t1 - t2).abs() < 0.01);
    }

    #[test]
    fn test_pca_mlmodel_roundtrip() {
        let x = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let inner = PcaModel::fit(&x, 1);
        let model = PcaMlModel {
            metadata: ModelMetadata {
                algorithm: Algorithm::PCA,
                num_features: 2,
                num_samples: 4,
                r_squared: None,
                mse: None,
                coefficients_count: 2,
                hyperparameters_json: "{}".into(),
            },
            inner,
        };

        let blob = model.serialize().unwrap();
        let recovered = PcaMlModel::deserialize(&blob).unwrap();
        assert_eq!(recovered.algorithm(), Algorithm::PCA);
        assert_eq!(recovered.inner.n_features(), 2);

        let orig = model.predict(&[4.0, 5.0]).unwrap();
        let recv = recovered.predict(&[4.0, 5.0]).unwrap();
        assert!((orig - recv).abs() < 0.01);
    }
}
