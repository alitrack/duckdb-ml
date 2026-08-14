//! LDA (Linear Discriminant Analysis) — supervised dimensionality reduction.
//!
//! Solves the generalized eigenproblem S_b v = λ S_w v (between/within-class
//! scatter) via: S_w + ridge → Cholesky L, symmetrize M = L⁻¹S_bL⁻ᵀ, extract
//! top-k eigenvectors by power iteration (deflated), back-transform
//! v = L⁻ᵀu. Mirrors sklearn's LinearDiscriminantAnalysis transform.

use crate::model::Algorithm;

/// LDA model: mean + discriminant directions (components[comp][feature]).
#[derive(Debug, Clone)]
pub struct LdaModel {
    mean: Vec<f64>,
    components: Vec<Vec<f64>>,
    eigenvalues: Vec<f64>,
}

impl LdaModel {
    /// Fit LDA. `n_components` is capped at min(n_classes−1, n_features).
    /// Returns None if fewer than 2 classes or a degenerate scatter matrix.
    pub fn fit(x: &[Vec<f64>], y: &[f64], n_components: usize) -> Option<LdaModel> {
        let n_samples = x.len();
        let n_features = x[0].len();
        if n_samples < 2 || n_features == 0 {
            return None;
        }
        // Group samples by class label (numeric labels → sorted unique ids).
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        if classes.len() < 2 {
            return None;
        }
        let class_id: Vec<usize> = y
            .iter()
            .map(|v| classes.iter().position(|c| c == v).unwrap())
            .collect();
        let n_classes = classes.len();
        let k = n_components.min(n_classes - 1).min(n_features);
        if k == 0 {
            return None;
        }

        // Global mean
        let mut mean = vec![0.0f64; n_features];
        for row in x {
            for j in 0..n_features {
                mean[j] += row[j];
            }
        }
        for v in mean.iter_mut() {
            *v /= n_samples as f64;
        }

        // Class means + sizes
        let mut class_means = vec![vec![0.0f64; n_features]; n_classes];
        let mut class_sizes = vec![0usize; n_classes];
        for (i, &c) in class_id.iter().enumerate() {
            class_sizes[c] += 1;
            for j in 0..n_features {
                class_means[c][j] += x[i][j];
            }
        }
        for c in 0..n_classes {
            let sz = class_sizes[c] as f64;
            for v in class_means[c].iter_mut() {
                *v /= sz;
            }
        }

        // Within-class scatter S_w (d×d)
        let mut sw = vec![0.0f64; n_features * n_features];
        for (i, &c) in class_id.iter().enumerate() {
            for j in 0..n_features {
                let dj = x[i][j] - class_means[c][j];
                for l in 0..n_features {
                    sw[j * n_features + l] += dj * (x[i][l] - class_means[c][l]);
                }
            }
        }

        // Between-class scatter S_b (d×d)
        let mut sb = vec![0.0f64; n_features * n_features];
        for c in 0..n_classes {
            let nc = class_sizes[c] as f64;
            for j in 0..n_features {
                let dj = class_means[c][j] - mean[j];
                for l in 0..n_features {
                    sb[j * n_features + l] += nc * dj * (class_means[c][l] - mean[l]);
                }
            }
        }

        // Ridge-regularized Cholesky of S_w: S_wr = L Lᵀ (L lower-triangular)
        let ridge = 1e-9 * {
            let tr: f64 = (0..n_features).map(|i| sw[i * n_features + i]).sum();
            (tr / n_features as f64).max(1e-12)
        };
        let mut l = vec![0.0f64; n_features * n_features]; // lower
        for i in 0..n_features {
            for j in 0..=i {
                let mut sum = sw[i * n_features + j];
                if i == j {
                    sum += ridge;
                }
                for m in 0..j {
                    sum -= l[i * n_features + m] * l[j * n_features + m];
                }
                let diag = if i == j { sum } else { l[j * n_features + j] };
                if i == j {
                    if sum <= 0.0 {
                        return None; // S_w not positive definite even with ridge
                    }
                    l[i * n_features + i] = sum.sqrt();
                } else {
                    l[i * n_features + j] = sum / diag;
                }
            }
        }

        // Symmetrized matrix M = L⁻¹ S_b L⁻ᵀ (symmetric)
        // Column j of M = L⁻¹ (S_b (L⁻ᵀ e_j))
        let mut m = vec![0.0f64; n_features * n_features];
        let mut tmp = vec![0.0f64; 2 * n_features]; // [0..nf) = v workspace, [nf..2nf) = w scratch
        for j in 0..n_features {
            // Solve Lᵀ v = e_j (upper-triangular, back substitution, k descending)
            // Lᵀ[k][m] = L[m][k] for m > k; v[k] = (e_j[k] − Σ_{m>k} Lᵀ[k][m] v[m]) / Lᵀ[k][k]
            for k in (0..n_features).rev() {
                let mut acc = if k == j { 1.0 } else { 0.0 };
                for m in (k + 1)..n_features {
                    acc -= l[m * n_features + k] * tmp[m];
                }
                tmp[k] = acc / l[k * n_features + k];
            }
            // y = S_b v
            let mut y = vec![0.0f64; n_features];
            for a in 0..n_features {
                for b in 0..n_features {
                    y[a] += sb[a * n_features + b] * tmp[b];
                }
            }
            // Solve L w = y (lower, forward substitution)
            for k in 0..n_features {
                let mut acc = y[k];
                for m in 0..k {
                    acc -= l[k * n_features + m] * tmp[n_features + m];
                }
                // reuse tmp[n_features..] as scratch for w
                tmp[n_features + k] = acc / l[k * n_features + k];
            }
            for k in 0..n_features {
                m[k * n_features + j] = tmp[n_features + k];
            }
        }

        // Power iteration (deflated) on M for top-k eigenvectors u
        let mut components = Vec::with_capacity(k);
        let mut eigenvalues = Vec::with_capacity(k);
        let mut msym = m.clone();
        for _ in 0..k {
            let u = power_iteration(&msym, n_features, &components);
            let ev = rayleigh_quotient(&msym, n_features, &u);
            // Back-transform: v = L⁻ᵀ u → solve Lᵀ v = u (upper-triangular back substitution)
            let mut v = vec![0.0f64; n_features];
            for kk in (0..n_features).rev() {
                let mut acc = u[kk];
                for m in (kk + 1)..n_features {
                    acc -= l[m * n_features + kk] * v[m];
                }
                v[kk] = acc / l[kk * n_features + kk];
            }
            // Normalize direction
            let norm: f64 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for a in v.iter_mut() {
                *a /= norm;
            }
            components.push(v);
            eigenvalues.push(ev);
            // Deflate M
            for a in 0..n_features {
                for b in 0..n_features {
                    msym[a * n_features + b] -= ev * u[a] * u[b];
                }
            }
        }

        Some(LdaModel {
            mean,
            components,
            eigenvalues,
        })
    }

    /// Project features onto the discriminant space.
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

    /// First discriminant score (MlModel-compatible single value).
    pub fn score(&self, features: &[f64]) -> f64 {
        self.transform(features).first().copied().unwrap_or(0.0)
    }

    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    pub fn n_features(&self) -> usize {
        self.mean.len()
    }

    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    // ——— Serialization (mirrors PcaModel format) ———
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
        for &v in &self.eigenvalues {
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
        let mut eigenvalues = Vec::with_capacity(nc);
        for _ in 0..nc {
            let b: [u8; 8] = data[pos..pos + 8].try_into().ok()?;
            eigenvalues.push(f64::from_le_bytes(b));
            pos += 8;
        }
        Some(Self {
            mean,
            components,
            eigenvalues,
        })
    }
}

/// Power iteration: dominant eigenvector of symmetric matrix (deflated).
fn power_iteration(a: &[f64], n: usize, existing: &[Vec<f64>]) -> Vec<f64> {
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..50 {
        for ev in existing {
            let dot: f64 = v.iter().zip(ev.iter()).map(|(&a, &b)| a * b).sum();
            for (vi, &ei) in v.iter_mut().zip(ev.iter()) {
                *vi -= dot * ei;
            }
        }
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[i * n + j] * v[j];
            }
        }
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

/// Rayleigh quotient: vᵀAv / vᵀv.
fn rayleigh_quotient(a: &[f64], n: usize, v: &[f64]) -> f64 {
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += a[i * n + j] * v[j];
        }
    }
    let num: f64 = av.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let den: f64 = v.iter().map(|a| a * a).sum();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// MlModel wrapper: predict returns the first discriminant score.
pub struct LdaMlModel {
    pub metadata: crate::model::ModelMetadata,
    pub inner: LdaModel,
}

impl crate::model::MlModel for LdaMlModel {
    fn predict(&self, features: &[f64]) -> Result<f64, crate::model::ModelError> {
        Ok(self.inner.score(features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::LDA
    }

    fn metadata(&self) -> &crate::model::ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, crate::model::ModelError> {
        Ok(self.inner.to_bytes())
    }

    fn deserialize(blob: &[u8]) -> Result<Self, crate::model::ModelError> {
        let inner = LdaModel::from_bytes(blob).ok_or_else(|| {
            crate::model::ModelError::Serialization("Failed to decode LDA model".into())
        })?;
        Ok(Self {
            metadata: crate::model::ModelMetadata {
                algorithm: Algorithm::LDA,
                num_features: inner.n_features(),
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: inner.n_components() * inner.n_features(),
                hyperparameters_json: serde_json::json!({
                    "n_components": inner.n_components(),
                    "eigenvalues": inner.eigenvalues(),
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

    fn two_gaussians() -> (Vec<Vec<f64>>, Vec<f64>) {
        // class 0: mean (0,0) gaussian-ish grid; class 1: mean (3,0)
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                x.push(vec![i as f64 / 5.0 - 2.0, j as f64 / 5.0 - 2.0]);
                y.push(0.0);
            }
        }
        for i in 0..20 {
            for j in 0..20 {
                x.push(vec![i as f64 / 5.0 + 1.0, j as f64 / 5.0 - 2.0]);
                y.push(1.0);
            }
        }
        (x, y)
    }

    #[test]
    fn separates_two_classes() {
        let (x, y) = two_gaussians();
        let model = LdaModel::fit(&x, &y, 1).unwrap();
        assert_eq!(model.n_components(), 1);
        // class means must be well separated along the discriminant
        let m0: Vec<f64> = (0..400)
            .filter(|&i| y[i] == 0.0)
            .map(|i| model.score(&x[i]))
            .collect();
        let m1: Vec<f64> = (400..800)
            .filter(|&i| y[i] == 1.0)
            .map(|i| model.score(&x[i]))
            .collect();
        let mean0: f64 = m0.iter().sum::<f64>() / m0.len() as f64;
        let mean1: f64 = m1.iter().sum::<f64>() / m1.len() as f64;
        assert!((mean1 - mean0).abs() > 1.0, "means should separate: {mean0} vs {mean1}");
    }

    #[test]
    fn caps_at_classes_minus_one() {
        let (x, y) = two_gaussians();
        let model = LdaModel::fit(&x, &y, 8).unwrap();
        assert_eq!(model.n_components(), 1); // min(2-1, 2, 8)
    }

    #[test]
    fn serialization_roundtrip() {
        let (x, y) = two_gaussians();
        let model = LdaModel::fit(&x, &y, 1).unwrap();
        let bytes = model.to_bytes();
        let back = LdaModel::from_bytes(&bytes).unwrap();
        let f = vec![0.5, 0.5];
        let p1 = model.transform(&f);
        let p2 = back.transform(&f);
        assert!((p1[0] - p2[0]).abs() < 1e-12);
    }

    #[test]
    fn rejects_single_class() {
        let (x, _) = two_gaussians();
        let y = vec![0.0; x.len()];
        assert!(LdaModel::fit(&x, &y, 1).is_none());
    }
}
