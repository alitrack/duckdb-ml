//! Support Vector Regression (ε-SVR) — hand-written SMO.
//!
//! PostgresML exposes SVR via smartcore; per the ElasticNet precedent the
//! crate's optimizer paths proved unreliable, so SVR is implemented directly:
//! ε-insensitive loss, dual solved by sequential minimal optimization with
//! deterministic full-pair sweeps. Supports linear / RBF / polynomial /
//! sigmoid kernels. Model format: custom blob (kernel type + params,
//! support vectors, coefficients β, bias) — the SVC family uses linfa-svm,
//! this module is independent.

use super::TrainingResult;
use std::error::Error;

/// Kernel type enum (stored in the blob).
#[derive(Debug, Clone, Copy, PartialEq, bincode::Encode, bincode::Decode)]
pub enum Kernel {
    Linear,
    Rbf,
    Polynomial,
    Sigmoid,
}

impl Kernel {
    pub fn parse(s: &str) -> Option<Kernel> {
        match s {
            "linear" => Some(Kernel::Linear),
            "rbf" | "gaussian" => Some(Kernel::Rbf),
            "poly" | "polynomial" => Some(Kernel::Polynomial),
            "sigmoid" => Some(Kernel::Sigmoid),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Kernel::Linear => "linear",
            Kernel::Rbf => "rbf",
            Kernel::Polynomial => "polynomial",
            Kernel::Sigmoid => "sigmoid",
        }
    }
}

/// Compute kernel value K(a, b).
fn kernel_value(k: Kernel, gamma: f64, degree: usize, coef0: f64, a: &[f64], b: &[f64]) -> f64 {
    match k {
        Kernel::Linear => a.iter().zip(b).map(|(x, y)| x * y).sum(),
        Kernel::Rbf => {
            let d2: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum();
            (-gamma * d2).exp()
        }
        Kernel::Polynomial => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            (gamma * dot + coef0).powi(degree as i32)
        }
        Kernel::Sigmoid => {
            (gamma * a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() + coef0).tanh()
        }
    }
}

/// ε-SVR training result (support vectors + coefficients + bias).
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub struct SvrModelData {
    pub kernel: Kernel,
    pub gamma: f64,
    pub degree: usize,
    pub coef0: f64,
    pub n_features: usize,
    pub bias: f64,
    /// (β_i, feature vector) per support vector, β_i = α_i − α*_i ∈ (−C, C)
    pub support: Vec<(f64, Vec<f64>)>,
    pub c: f64,
    pub epsilon: f64,
    pub r_squared: Option<f64>,
    pub mse: Option<f64>,
}

/// Train ε-SVR.
///
/// `c` = box constraint (default 1.0), `epsilon` = tube width (default 0.1),
/// `gamma` = kernel width for rbf/poly/sigmoid (default 1/n_features),
/// `degree`/`coef0` for poly/sigmoid, `tol` = SMO sweep tolerance,
/// `max_iter` = sweep cap.
#[allow(clippy::too_many_arguments)]
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    kernel: Kernel,
    c: f64,
    epsilon: f64,
    gamma: f64,
    degree: usize,
    coef0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<TrainingResult, Box<dyn Error>> {
    let n = x.len();
    if n == 0 {
        return Err("svr: empty training data".into());
    }
    if c <= 0.0 {
        return Err("svr: c must be > 0".into());
    }
    if epsilon < 0.0 {
        return Err("svr: epsilon must be >= 0".into());
    }
    if degree == 0 {
        return Err("svr: degree must be >= 1".into());
    }
    let n_features = x[0].len();
    if x.iter().any(|s| s.len() != n_features) {
        return Err("svr: inconsistent feature dimensions".into());
    }
    if y.len() != n {
        return Err("svr: x/y length mismatch".into());
    }
    if n > 1000 {
        return Err("svr: dataset too large for kernel matrix (max 1000)".into());
    }
    let gamma = if gamma > 0.0 {
        gamma
    } else {
        1.0 / n_features as f64
    };

    // precompute kernel matrix
    let mut k = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = kernel_value(kernel, gamma, degree, coef0, &x[i], &x[j]);
        }
    }

    let mut beta = vec![0.0f64; n];
    let mut bias = 0.0f64;
    let mut e = vec![0.0f64; n]; // unbiased gradient g_i = (Kβ)_i − y_i, bias held 0 during iteration
    for i in 0..n {
        e[i] = -y[i];
    }

    // Working-set SMO (libsvm-style): each round picks the single most
    // violating pair by directional gradients and updates it; gradients are
    // maintained incrementally (O(n) per round instead of O(n²) full sweep).
    // min-form objective: min ½βᵀKβ − yᵀβ + εΣ|β|, s.t. Σβ=0, β∈[−C,C]
    // d_i = ∂/∂βᵢ = g_i + ε·sign(β_i). i = argmin d over β_i<C (increase),
    // j = argmax d over β_j>−C (decrease); converged when d_j − d_i ≤ tol.
    let sgn = |v: f64| -> f64 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    for _ in 0..max_iter {
        let mut dmin = f64::INFINITY;
        let mut ii = 0usize;
        let mut dmax = -f64::INFINITY;
        let mut jj = 0usize;
        for i in 0..n {
            let d = e[i] + epsilon * sgn(beta[i]);
            if beta[i] < c - 1e-9 && d < dmin {
                dmin = d;
                ii = i;
            }
            if beta[i] > -c + 1e-9 && d > dmax {
                dmax = d;
                jj = i;
            }
        }
        if jj == ii {
            // degenerate: same point wins both sides (all free gradients equal
            // or n==1) — no improving pair exists
            break;
        }
        if dmax - dmin < tol {
            break;
        }
        let eta = k[ii * n + ii] + k[jj * n + jj] - 2.0 * k[ii * n + jj];
        if eta <= 1e-12 {
            break; // degenerate kernel slice; cannot improve further
        }
        // closed-form 2-variable minimizer: Δβ = (d_j − d_i)/η, clamped to box
        let num = e[jj] - e[ii] + epsilon * (sgn(beta[jj]) - sgn(beta[ii]));
        let delta = num / eta;
        let s = beta[ii] + beta[jj];
        let lo = (-c).max(s - c);
        let hi = c.min(s + c);
        let new_i = (beta[ii] + delta).clamp(lo, hi);
        let new_j = s - new_i;
        if (new_i - beta[ii]).abs() < 1e-14 {
            break; // no movement; further rounds cannot improve
        }
        let di = new_i - beta[ii];
        let dj = new_j - beta[jj];
        beta[ii] = new_i;
        beta[jj] = new_j;
        // incremental gradient update: g_k += Δβ_i·K(k,i) + Δβ_j·K(k,j)
        for kk in 0..n {
            e[kk] += di * k[kk * n + ii] + dj * k[kk * n + jj];
        }
    }

    // final bias: average over free support vectors (0 < |β| < C).
    // e_i = (Kβ)_i − y_i  ⇒  b = y_i − (Kβ)_i − ε·sign(β_i) = −e_i − ε·sign(β_i)
    let mut b_sum = 0.0f64;
    let mut b_cnt = 0usize;
    for i in 0..n {
        if beta[i].abs() > 1e-9 && beta[i].abs() < c - 1e-9 {
            let sign = if beta[i] > 0.0 { 1.0 } else { -1.0 };
            b_sum += -e[i] - epsilon * sign;
            b_cnt += 1;
        }
    }
    if b_cnt > 0 {
        bias = b_sum / b_cnt as f64;
    }

    // keep only support vectors
    let support: Vec<(f64, Vec<f64>)> = (0..n)
        .filter(|&i| beta[i].abs() > 1e-9)
        .map(|i| (beta[i], x[i].clone()))
        .collect();

    let data = SvrModelData {
        kernel,
        gamma,
        degree,
        coef0,
        n_features,
        bias,
        support,
        c,
        epsilon,
        r_squared: None,
        mse: None,
    };

    // metrics on the training set
    let mut predictions = Vec::with_capacity(n);
    for row in x {
        predictions.push(predict_one(&data, row));
    }
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = predictions
        .iter()
        .zip(y.iter())
        .map(|(p, a)| (a - p).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|a| (a - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 1e-10 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    };
    let mse = Some(ss_res / n as f64);

    let mut data = data;
    data.r_squared = r_squared;
    data.mse = mse;
    let blob = serialize(&data)?;

    Ok(TrainingResult {
        coefficients: vec![],
        intercept: bias,
        r_squared,
        mse,
        num_samples: n,
        model_blob: Some(blob),
    })
}

/// Predict for one feature vector.
pub fn predict_one(d: &SvrModelData, features: &[f64]) -> f64 {
    let mut acc = d.bias;
    for (b, sv) in &d.support {
        acc += b * kernel_value(d.kernel, d.gamma, d.degree, d.coef0, sv, features);
    }
    acc
}

/// Serialize SVR model (bincode of `SvrModelData`).
pub fn serialize(d: &SvrModelData) -> Result<Vec<u8>, String> {
    bincode::encode_to_vec(d, bincode::config::standard()).map_err(|e| format!("svr: {e}"))
}

/// Deserialize SVR model from blob.
pub fn deserialize(blob: &[u8]) -> Result<SvrModelData, String> {
    bincode::decode_from_slice(blob, bincode::config::standard())
        .map(|(d, _)| d)
        .map_err(|e| format!("svr: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_svr_fits_line() {
        // y = 3 + 2x, epsilon=0 → nearly exact
        let x: Vec<Vec<f64>> = (0..14).map(|i| vec![i as f64 * 0.5]).collect();
        let y: Vec<f64> = (0..14).map(|i| 3.0 + 2.0 * i as f64 * 0.5).collect();
        let r = train(&x, &y, Kernel::Linear, 1.0, 1e-6, 0.0, 2, 0.0, 1e-4, 2000).unwrap();
        let d = deserialize(r.model_blob.as_ref().unwrap()).unwrap();
        let p = predict_one(&d, &[7.0]);
        assert!((p - (3.0 + 2.0 * 7.0)).abs() < 0.1, "pred={p}");
        assert!(!d.support.is_empty());
    }

    #[test]
    fn rbf_svr_fits_curve() {
        // y = x² on [0,3]; RBF kernel interpolates
        let x: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64 * 0.25]).collect();
        let y: Vec<f64> = x.iter().map(|s| s[0] * s[0]).collect();
        let r = train(&x, &y, Kernel::Rbf, 100.0, 1e-4, 2.0, 2, 0.0, 1e-4, 3000).unwrap();
        let d = deserialize(r.model_blob.as_ref().unwrap()).unwrap();
        let p = predict_one(&d, &[1.5]);
        assert!((p - 2.25).abs() < 0.1, "pred={p}");
    }

    #[test]
    fn poly_kernel_quadratic() {
        // degree-2 poly kernel reproduces quadratic exactly
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|s| 1.0 + 2.0 * s[0] + 3.0 * s[0] * s[0])
            .collect();
        let r = train(
            &x,
            &y,
            Kernel::Polynomial,
            100.0,
            1e-4,
            1.0,
            2,
            1.0,
            1e-4,
            3000,
        )
        .unwrap();
        let d = deserialize(r.model_blob.as_ref().unwrap()).unwrap();
        let p = predict_one(&d, &[1.0]);
        assert!((p - 6.0).abs() < 0.2, "pred={p}");
    }

    #[test]
    fn noisy_data_stays_in_tube() {
        // linear data with mild noise: most points inside the ε-tube
        // (noise amplitude < ε), so SVs are sparse
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20)
            .map(|i| 2.0 * i as f64 + (i as f64 * 7.3).sin() * 0.2)
            .collect();
        let r = train(&x, &y, Kernel::Linear, 50.0, 0.3, 0.0, 2, 0.0, 1e-4, 4000).unwrap();
        let d = deserialize(r.model_blob.as_ref().unwrap()).unwrap();
        let p = predict_one(&d, &[10.0]);
        assert!((p - 20.0).abs() < 0.8, "pred={p}");
        assert!(
            d.support.len() < 20,
            "expected sparse SVs, got {}",
            d.support.len()
        );
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[], &[], Kernel::Linear, 1.0, 0.1, 0.0, 2, 0.0, 1e-3, 1000).is_err());
        assert!(train(
            &[vec![1.0]],
            &[1.0],
            Kernel::Linear,
            0.0,
            0.1,
            0.0,
            2,
            0.0,
            1e-3,
            1000
        )
        .is_err());
        assert!(train(
            &[vec![1.0]],
            &[1.0],
            Kernel::Linear,
            1.0,
            -0.1,
            0.0,
            2,
            0.0,
            1e-3,
            1000
        )
        .is_err());
        assert!(train(
            &[vec![1.0, 2.0], vec![1.0]],
            &[1.0, 2.0],
            Kernel::Linear,
            1.0,
            0.1,
            0.0,
            2,
            0.0,
            1e-3,
            1000
        )
        .is_err());
        assert!(train(
            &[vec![1.0]],
            &[1.0, 2.0],
            Kernel::Linear,
            1.0,
            0.1,
            0.0,
            2,
            0.0,
            1e-3,
            1000
        )
        .is_err());
        assert!(Kernel::parse("rbf").is_some());
        assert!(Kernel::parse("bogus").is_none());
    }

    #[test]
    fn blob_roundtrip() {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let y: Vec<f64> = (0..8).map(|i| 1.0 + 0.5 * i as f64).collect();
        let r = train(&x, &y, Kernel::Rbf, 1.0, 0.1, 0.5, 2, 0.0, 1e-3, 1000).unwrap();
        let d1 = deserialize(r.model_blob.as_ref().unwrap()).unwrap();
        let blob2 = serialize(&d1).unwrap();
        let d2 = deserialize(&blob2).unwrap();
        assert_eq!(d1.support.len(), d2.support.len());
        assert_eq!(d1.bias, d2.bias);
        assert_eq!(d1.kernel, d2.kernel);
    }
}
