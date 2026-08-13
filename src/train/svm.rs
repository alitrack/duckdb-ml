//! SVM binary classification via linfa-svm (0.8.1, MIT — libsvm SMO core).
//!
//! Trains a binary SVC with linear or Gaussian (RBF) kernel. The trained
//! `Svm<f64, bool>` (support vectors, dual coefficients, kernel params, bias)
//! is serialized with bincode's serde layer into the model blob; prediction
//! reuses linfa's own `Predict` impl, so training and inference semantics are
//! identical to upstream libsvm.

use linfa::dataset::Dataset;
use linfa::traits::{Fit, Predict};
use linfa_svm::Svm;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Kernel selector encoded in the f64 params map:
/// 0 = linear, 1 = Gaussian/RBF (default).
pub const KERNEL_LINEAR: u8 = 0;
pub const KERNEL_GAUSSIAN: u8 = 1;

/// Self-describing blob: feature count + kernel ride along with the SVM so
/// deserialization needs no external context.
#[derive(Serialize, Deserialize)]
pub struct SvmBlob {
    pub n_features: usize,
    pub kernel: u8,
    pub svm: Svm<f64, bool>,
}

/// Trained SVM + metadata that survives (de)serialization.
pub struct SvmTrained {
    /// bincode(serde) blob of the `SvmBlob`.
    pub blob: Vec<u8>,
    /// Number of support vectors with |alpha| > 100·ε.
    pub n_support: usize,
    /// Decision threshold offset (rho).
    pub rho: f64,
    pub n_features: usize,
    pub kernel: u8,
}

/// Train a binary SVM. `y` must contain only 0.0 / 1.0 labels.
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    c: f64,
    kernel: u8,
    gamma: f64,
) -> Result<SvmTrained, String> {
    if x.is_empty() || x.len() != y.len() {
        return Err("empty or mismatched data".into());
    }
    let n_features = x[0].len();
    if n_features == 0 {
        return Err("zero features".into());
    }
    if c <= 0.0 {
        return Err("c must be > 0".into());
    }
    if gamma <= 0.0 {
        return Err("gamma must be > 0".into());
    }
    if kernel != KERNEL_LINEAR && kernel != KERNEL_GAUSSIAN {
        return Err(format!("unknown kernel {kernel} (0=linear, 1=gaussian)"));
    }

    let flat: Vec<f64> = x.iter().flatten().copied().collect();
    let records: Array2<f64> = ndarray::Array2::from_shape_vec((x.len(), n_features), flat)
        .map_err(|e| format!("array build: {e}"))?;

    let targets: Vec<bool> = y
        .iter()
        .map(|v| {
            if *v == 0.0 {
                Ok(false)
            } else if *v == 1.0 {
                Ok(true)
            } else {
                Err(format!("labels must be 0/1, got {v}"))
            }
        })
        .collect::<Result<Vec<bool>, String>>()?;

    let dataset = Dataset::new(records, ndarray::Array1::from(targets));

    let svm: Svm<f64, bool> = {
        let p = Svm::<f64, bool>::params().pos_neg_weights(c, c);
        match kernel {
            KERNEL_LINEAR => p.linear_kernel(),
            _ => p.gaussian_kernel(gamma),
        }
    }
    .fit(&dataset)
    .map_err(|e| format!("training failed: {e}"))?;

    let n_support = svm.nsupport();
    let rho = svm.rho;
    let blob_obj = SvmBlob {
        n_features,
        kernel,
        svm,
    };
    let blob = bincode::serde::encode_to_vec(&blob_obj, bincode::config::standard())
        .map_err(|e| format!("serialize: {e}"))?;

    Ok(SvmTrained {
        blob,
        n_support,
        rho,
        n_features,
        kernel,
    })
}

/// Deserialize a trained SVM blob.
pub fn deserialize(blob: &[u8]) -> Result<SvmBlob, String> {
    bincode::serde::decode_from_slice(blob, bincode::config::standard())
        .map(|(b, _)| b)
        .map_err(|e| format!("deserialize: {e}"))
}

/// Predict the class (0.0 / 1.0) for one feature row.
pub fn predict_one(svm: &Svm<f64, bool>, features: &[f64]) -> f64 {
    let row = ndarray::Array1::from(features.to_vec());
    if svm.predict(row) {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two linearly separable blobs: x < 0 → class 0, x > 0 → class 1.
    fn separable_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = vec![
            vec![-1.0, -1.0],
            vec![-0.8, -0.9],
            vec![-1.2, -1.1],
            vec![-0.9, -0.7],
            vec![1.0, 1.0],
            vec![1.2, 0.9],
            vec![0.8, 1.1],
            vec![1.1, 1.2],
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn linear_separable() {
        let (x, y) = separable_data();
        let t = train(&x, &y, 1.0, KERNEL_LINEAR, 1.0).unwrap();
        let blob = deserialize(&t.blob).unwrap();
        assert!(t.n_support > 0);
        assert_eq!(blob.n_features, 2);
        assert_eq!(blob.kernel, KERNEL_LINEAR);
        // exact separation
        assert_eq!(predict_one(&blob.svm, &[-0.7, -0.8]), 0.0);
        assert_eq!(predict_one(&blob.svm, &[0.9, 1.0]), 1.0);
        assert_eq!(predict_one(&blob.svm, &[-0.1, -0.1]), 0.0);
        assert_eq!(predict_one(&blob.svm, &[0.1, 0.1]), 1.0);
    }

    #[test]
    fn gaussian_nonlinear_ring() {
        // circle: inside radius ~1 → 1, outside → 0 (needs RBF)
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..60 {
            let a = i as f64 * std::f64::consts::TAU / 60.0;
            // inner ring
            x.push(vec![0.3 * a.cos(), 0.3 * a.sin()]);
            y.push(1.0);
            // outer ring
            x.push(vec![2.0 * a.cos(), 2.0 * a.sin()]);
            y.push(0.0);
        }
        let t = train(&x, &y, 1.0, KERNEL_GAUSSIAN, 0.5).unwrap();
        let blob = deserialize(&t.blob).unwrap();
        assert!(predict_one(&blob.svm, &[0.0, 0.0]) > 0.5);
        assert!(predict_one(&blob.svm, &[1.5, 0.0]) < 0.5);
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[], &[], 1.0, KERNEL_LINEAR, 1.0).is_err());
        assert!(train(&[vec![1.0]], &[0.0, 1.0], 1.0, KERNEL_LINEAR, 1.0).is_err());
        assert!(train(&[vec![1.0]], &[0.5], 1.0, KERNEL_LINEAR, 1.0).is_err());
        assert!(train(&[vec![1.0]], &[0.0], -1.0, KERNEL_LINEAR, 1.0).is_err());
        assert!(train(&[vec![1.0]], &[0.0], 1.0, 7, 1.0).is_err());
        assert!(deserialize(&[0u8, 1, 2]).is_err());
    }

    #[test]
    fn blob_roundtrip_stable() {
        let (x, y) = separable_data();
        let t1 = train(&x, &y, 1.0, KERNEL_LINEAR, 1.0).unwrap();
        let t2 = train(&x, &y, 1.0, KERNEL_LINEAR, 1.0).unwrap();
        assert_eq!(t1.blob, t2.blob); // deterministic training
    }
}
