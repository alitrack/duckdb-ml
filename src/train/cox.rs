//! Cox proportional hazards regression — MADlib `cox_prop_hazards`
//! counterpart (partial likelihood, Breslow tie handling).
//!
//! h(t|x) = h0(t)·exp(w·x); negative log partial likelihood
//! NLL = −Σ_{i:e_i=1} [w·x_i − log Σ_{j∈R(t_i)} exp(w·x_j)]
//! optimized by full-batch gradient ascent on the partial log-likelihood.
//! `predict` returns the relative risk exp(w·x) (MADlib `cox_predict`).
//!
//! Blob layout: u32 d · d×f64 weights

/// Trained Cox model.
pub struct CoxModel {
    pub weights: Vec<f64>,
    pub n_features: usize,
}

/// `time` > 0 event times; `event` ∈ {0,1} censoring flags.
pub fn train(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[f64],
    lr: f64,
    max_epochs: usize,
) -> Result<CoxModel, String> {
    if x.is_empty() || x.len() != time.len() || time.len() != event.len() {
        return Err("empty or mismatched data".into());
    }
    let n = x.len();
    let d = x[0].len();
    if d == 0 {
        return Err("zero features".into());
    }
    if lr <= 0.0 {
        return Err("lr must be > 0".into());
    }
    for t in time {
        if *t <= 0.0 {
            return Err(format!("time must be > 0, got {t}"));
        }
    }
    let ev: Vec<bool> = event
        .iter()
        .map(|e| {
            if *e == 0.0 {
                Ok(false)
            } else if *e == 1.0 {
                Ok(true)
            } else {
                Err(format!("event must be 0/1, got {e}"))
            }
        })
        .collect::<Result<Vec<bool>, String>>()?;

    // sort descending by time so the risk set is a growing suffix
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut w = vec![0.0f64; d];
    let mut prev_nll = f64::MAX;

    for _epoch in 0..max_epochs {
        // suffix accumulators over risk set (all with time >= current)
        // risk_sum = Σ exp(w·x), risk_x[j] = Σ exp(w·x)·x_j
        let mut risk_sum = 0.0f64;
        let mut risk_x = vec![0.0f64; d];
        let mut grad = vec![0.0f64; d];
        let mut nll = 0.0f64;

        for &i in &order {
            // add sample i to the risk set first (ties share the set: Breslow)
            let dot = w.iter().zip(&x[i]).map(|(a, b)| a * b).sum::<f64>();
            let ex = dot.exp().min(f64::MAX / 2.0);
            risk_sum += ex;
            for j in 0..d {
                risk_x[j] += ex * x[i][j];
            }
            if ev[i] {
                // contribution of event i
                let denom = risk_sum.max(1e-300);
                nll -= dot - denom.ln();
                for j in 0..d {
                    grad[j] -= x[i][j] - risk_x[j] / denom;
                }
            }
        }

        for g in grad.iter_mut() {
            *g /= n as f64;
        }
        for j in 0..d {
            w[j] -= lr * grad[j];
        }

        if (prev_nll - nll).abs() < 1e-6 {
            break;
        }
        prev_nll = nll;
    }

    Ok(CoxModel {
        weights: w,
        n_features: d,
    })
}

/// Relative risk exp(w·x).
pub fn predict_one(m: &CoxModel, features: &[f64]) -> f64 {
    let dot = m
        .weights
        .iter()
        .zip(features)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    dot.exp()
}

pub fn serialize(m: &CoxModel) -> Vec<u8> {
    let d = m.n_features;
    let mut out = Vec::with_capacity(4 + d * 8);
    out.extend_from_slice(&(d as u32).to_le_bytes());
    for w in &m.weights {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

pub fn deserialize(blob: &[u8]) -> Result<CoxModel, String> {
    if blob.len() < 4 {
        return Err("truncated blob".into());
    }
    let d = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    if d == 0 || blob.len() < 4 + d * 8 {
        return Err("truncated blob".into());
    }
    let mut weights = Vec::with_capacity(d);
    for i in 0..d {
        let mut b = [0u8; 8];
        b.copy_from_slice(&blob[4 + i * 8..4 + i * 8 + 8]);
        weights.push(f64::from_le_bytes(b));
    }
    Ok(CoxModel {
        weights,
        n_features: d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn treatment_effect_sign() {
        // treated (x=1) has higher hazard → positive coefficient
        let x = vec![
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![0.0],
        ];
        let time = vec![3.0, 5.0, 6.0, 7.0, 9.0, 4.0, 8.0, 9.0, 10.0, 11.0];
        let event = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let m = train(&x, &time, &event, 0.05, 3000).unwrap();
        // treated mostly die earlier → w > 0
        assert!(m.weights[0] > 0.05, "w = {}", m.weights[0]);
        // relative risk ordering
        assert!(predict_one(&m, &[1.0]) > predict_one(&m, &[0.0]));
    }

    #[test]
    fn censoring_ignored_events() {
        // all censored → no gradient signal → weights stay ~0
        let x = vec![vec![1.0], vec![0.0]];
        let time = vec![5.0, 6.0];
        let event = vec![0.0, 0.0];
        let m = train(&x, &time, &event, 0.1, 200).unwrap();
        assert!(m.weights[0].abs() < 1e-9);
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[], &[], &[], 0.1, 100).is_err());
        assert!(train(&[vec![1.0]], &[-1.0], &[1.0], 0.1, 100).is_err()); // time ≤ 0
        assert!(train(&[vec![1.0]], &[1.0], &[2.0], 0.1, 100).is_err()); // event = 2
        assert!(train(&[vec![1.0]], &[1.0], &[1.0], -0.1, 100).is_err());
        assert!(deserialize(&[0u8; 3]).is_err());
    }

    #[test]
    fn roundtrip() {
        let x = vec![vec![1.0], vec![0.0]];
        let time = vec![2.0, 3.0];
        let event = vec![1.0, 1.0];
        let m1 = train(&x, &time, &event, 0.05, 500).unwrap();
        let m2 = deserialize(&serialize(&m1)).unwrap();
        assert_eq!(serialize(&m1), serialize(&m2));
        assert_eq!(predict_one(&m2, &[1.0]), predict_one(&m1, &[1.0]));
    }
}
