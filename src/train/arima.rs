//! ARIMA(p,d,q) time-series model — MADlib `arima` counterpart.
//!
//! Apply d-th order differencing, then fit ARMA(p,q) with intercept by
//! conditional least squares: residuals are computed recursively
//!   ε_t = w_t − c − Σ φ_j (w_{t−j} − c) − Σ θ_m ε_{t−m}
//! and [c, φ, θ] are optimized by gradient descent with central-difference
//! numerical gradients (small p+q, tiny cost). `predict([h])` rolls the
//! recursion h steps beyond the training window (future ε = 0) and reverses
//! the differencing. Deterministic.
//!
//! Blob layout: u32 p · u32 d · u32 q · f64 intercept · p×f64 φ · q×f64 θ ·
//! last max(p,1) differenced values · last d raw values

/// Trained ARIMA model.
pub struct ArimaModel {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    /// AR coefficients φ_1..φ_p.
    pub ar: Vec<f64>,
    /// MA coefficients θ_1..θ_q.
    pub ma: Vec<f64>,
    /// Fitted intercept c (steady-state level of the differenced series).
    pub intercept: f64,
    /// Last max(p,1) differenced values (most recent last).
    pub diff_tail: Vec<f64>,
    /// Last d raw values (most recent last) for differencing reversal.
    pub raw_tail: Vec<f64>,
}

fn differenced(y: &[f64], d: usize) -> Vec<f64> {
    let mut cur = y.to_vec();
    for _ in 0..d {
        let next: Vec<f64> = cur.windows(2).map(|w| w[1] - w[0]).collect();
        cur = next;
    }
    cur
}

/// Loss = Σ ε_t² over the differenced series (conditional least squares).
/// Non-centered form: w_t = c + Σ φ_j w_{t−j} + Σ θ_m ε_{t−m} + ε_t.
fn residual_loss(series: &[f64], c: f64, ar: &[f64], ma: &[f64]) -> f64 {
    let p = ar.len();
    let q = ma.len();
    let n = series.len();
    let mut eps = vec![0.0f64; n];
    let mut loss = 0.0f64;
    for t in 0..n {
        let mut e = series[t] - c;
        for j in 0..p {
            if t > j {
                e -= ar[j] * series[t - j - 1];
            }
        }
        for m in 0..q {
            if t > m {
                e -= ma[m] * eps[t - m - 1];
            }
        }
        eps[t] = e;
        loss += e * e;
    }
    loss
}

/// Train ARIMA(p,d,q) on a time series `y` (length ≥ p+q+d+2).
pub fn train(
    y: &[f64],
    p: usize,
    d: usize,
    q: usize,
    lr: f64,
    max_epochs: usize,
) -> Result<ArimaModel, String> {
    if y.len() < p + q + d + 2 {
        return Err(format!(
            "series too short: {} points for p={p} d={d} q={q}",
            y.len()
        ));
    }
    if lr <= 0.0 {
        return Err("lr must be > 0".into());
    }
    let series = differenced(y, d);
    if series.len() < 2 {
        return Err("series too short after differencing".into());
    }

    // Closed-form conditional least squares for pure AR (q = 0): fit
    // w_t = c + Σ φ_j w_{t−j} via normal equations. Exact, no tuning.
    let mut c;
    let mut ar;
    let mut ma = vec![0.0f64; q];
    if q == 0 && p >= 1 && series.len() > p {
        let np = p + 1;
        let mut xtx = vec![0.0f64; np * np];
        let mut xty = vec![0.0f64; np];
        for t in p..series.len() {
            let mut row = vec![1.0f64];
            for j in 1..=p {
                row.push(series[t - j]);
            }
            for a in 0..np {
                for b in 0..np {
                    xtx[a * np + b] += row[a] * row[b];
                }
                xty[a] += row[a] * series[t];
            }
        }
        // Gaussian elimination with partial pivoting
        let mut a = xtx;
        let mut b = xty;
        let mut singular = false;
        for col in 0..np {
            let mut piv = col;
            for r in col + 1..np {
                if a[r * np + col].abs() > a[piv * np + col].abs() {
                    piv = r;
                }
            }
            if a[piv * np + col].abs() < 1e-12 {
                // degenerate design (e.g. constant series after differencing):
                // no AR structure to estimate → φ = 0, c = series mean
                singular = true;
                break;
            }
            if piv != col {
                for k in 0..np {
                    a.swap(piv * np + k, col * np + k);
                }
                b.swap(piv, col);
            }
            for r in col + 1..np {
                let f = a[r * np + col] / a[col * np + col];
                if f != 0.0 {
                    for k in col..np {
                        a[r * np + k] -= f * a[col * np + k];
                    }
                    b[r] -= f * b[col];
                }
            }
        }
        if singular {
            c = series.iter().sum::<f64>() / series.len() as f64;
            ar = vec![0.0f64; p];
        } else {
            let mut beta = vec![0.0f64; np];
            for col in (0..np).rev() {
                let mut s = b[col];
                for k in col + 1..np {
                    s -= a[col * np + k] * beta[k];
                }
                beta[col] = s / a[col * np + col];
            }
            c = beta[0];
            ar = beta[1..].to_vec();
        }
    } else {
        // general ARMA(p,q): gradient descent on conditional least squares
        c = series.iter().sum::<f64>() / series.len() as f64; // warm start
        ar = vec![0.0f64; p];
    }

    if q > 0 {
        // GD path for ARMA: [c, φ, θ] with central-difference gradients
        let np = 1 + p + q;
        let mut prev = f64::MAX;
        let params = |c: f64, ar: &[f64], ma: &[f64]| -> Vec<f64> {
            let mut v = Vec::with_capacity(np);
            v.push(c);
            v.extend_from_slice(ar);
            v.extend_from_slice(ma);
            v
        };
        for _epoch in 0..max_epochs {
            let loss = residual_loss(&series, c, &ar, &ma);
            let mut grad = vec![0.0f64; np];
            let cur = params(c, &ar, &ma);
            let h = 1e-6;
            for i in 0..np {
                let mut up = cur.clone();
                let mut dn = cur.clone();
                up[i] += h;
                dn[i] -= h;
                let l_up = residual_loss(&series, up[0], &up[1..1 + p], &up[1 + p..]);
                let l_dn = residual_loss(&series, dn[0], &dn[1..1 + p], &dn[1 + p..]);
                grad[i] = (l_up - l_dn) / (2.0 * h);
            }
            let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm > 1e-12 {
                c -= lr * grad[0] / norm;
                for i in 0..p {
                    ar[i] -= lr * grad[1 + i] / norm;
                }
                for i in 0..q {
                    ma[i] -= lr * grad[1 + p + i] / norm;
                }
            }
            if (prev - loss).abs() < 1e-9 {
                break;
            }
            prev = loss;
        }
    }

    let diff_tail: Vec<f64> = series[series.len().saturating_sub(p.max(1))..].to_vec();
    let raw_tail: Vec<f64> = y[y.len().saturating_sub(d)..].to_vec();

    Ok(ArimaModel {
        p,
        d,
        q,
        ar,
        ma,
        intercept: c,
        diff_tail,
        raw_tail,
    })
}

/// Forecast h steps beyond the training window (h ≥ 1).
pub fn forecast(m: &ArimaModel, h: usize) -> f64 {
    let p = m.p;
    let q = m.q;
    let c = m.intercept;
    // work on a copy of the diff tail, extended by h steps
    let mut w: Vec<f64> = m.diff_tail.clone();
    // future residuals are 0; we keep the past residuals in a ring buffer
    let ring_len = q.max(1);
    let mut eps = vec![0.0f64; ring_len];
    let start = w.len();
    for _step in 0..h {
        let mut val = c;
        // AR part uses w[-1], w[-2], ...
        for j in 0..p {
            let idx = w.len() as isize - 1 - j as isize;
            if idx >= 0 {
                val += m.ar[j] * w[idx as usize];
            }
        }
        // MA part uses past residuals (zeros for the future)
        for mm in 0..q {
            let idx = w.len() as isize - 1 - mm as isize;
            if idx >= 0 && idx < start as isize {
                // residual at that past step: recompute for correctness
                let t = idx as usize;
                let mut e = w[t] - c;
                for j in 0..p {
                    let jdx = t as isize - 1 - j as isize;
                    if jdx >= 0 {
                        e -= m.ar[j] * w[jdx as usize];
                    }
                }
                for kk in 0..q {
                    let kdx = t as isize - 1 - kk as isize;
                    if kdx >= 0 {
                        let prev_e = eps[kdx as usize % ring_len];
                        e -= m.ma[kk] * prev_e;
                    }
                }
                eps[t % ring_len] = e;
                val += m.ma[mm] * e;
            }
        }
        w.push(val);
        eps[(w.len() - 1) % ring_len] = 0.0;
    }
    // undo differencing: y_{t+1} = y_t + Δy_t with Δy updated per d
    let mut raw = m.raw_tail.clone();
    let mut delta = if raw.len() >= 2 {
        raw[raw.len() - 1] - raw[raw.len() - 2]
    } else {
        0.0
    };
    for &v in &w[start..] {
        if m.d >= 1 {
            if m.d == 1 {
                delta = v;
            } else {
                delta += v; // d ≥ 2: accumulate differences
            }
            let last = *raw.last().unwrap_or(&0.0);
            raw.push(last + delta);
        } else {
            raw.push(v);
        }
    }
    *raw.last().unwrap()
}

pub fn serialize(m: &ArimaModel) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(m.p as u32).to_le_bytes());
    out.extend_from_slice(&(m.d as u32).to_le_bytes());
    out.extend_from_slice(&(m.q as u32).to_le_bytes());
    for v in &m.ar {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &m.ma {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&m.intercept.to_le_bytes());
    out.extend_from_slice(&(m.diff_tail.len() as u32).to_le_bytes());
    for v in &m.diff_tail {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(m.raw_tail.len() as u32).to_le_bytes());
    for v in &m.raw_tail {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn deserialize(blob: &[u8]) -> Result<ArimaModel, String> {
    let mut off = 0usize;
    let rd_u32 = |off: &mut usize| -> Result<usize, String> {
        if blob.len() < *off + 4 {
            return Err("truncated blob".into());
        }
        let v = u32::from_le_bytes(blob[*off..*off + 4].try_into().unwrap()) as usize;
        *off += 4;
        Ok(v)
    };
    let rd_f64 = |off: &mut usize| -> Result<f64, String> {
        if blob.len() < *off + 8 {
            return Err("truncated blob".into());
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&blob[*off..*off + 8]);
        *off += 8;
        Ok(f64::from_le_bytes(b))
    };
    let p = rd_u32(&mut off)?;
    let d = rd_u32(&mut off)?;
    let q = rd_u32(&mut off)?;
    let mut ar = Vec::with_capacity(p);
    for _ in 0..p {
        ar.push(rd_f64(&mut off)?);
    }
    let mut ma = Vec::with_capacity(q);
    for _ in 0..q {
        ma.push(rd_f64(&mut off)?);
    }
    let intercept = rd_f64(&mut off)?;
    let dt_len = rd_u32(&mut off)?;
    let mut diff_tail = Vec::with_capacity(dt_len);
    for _ in 0..dt_len {
        diff_tail.push(rd_f64(&mut off)?);
    }
    let rt_len = rd_u32(&mut off)?;
    let mut raw_tail = Vec::with_capacity(rt_len);
    for _ in 0..rt_len {
        raw_tail.push(rd_f64(&mut off)?);
    }
    Ok(ArimaModel {
        p,
        d,
        q,
        ar,
        ma,
        intercept,
        diff_tail,
        raw_tail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_trend_ar1() {
        // y_t = 10 + 2t  → d=1 makes it constant → AR(1) φ→0, forecast exact
        let y: Vec<f64> = (0..30).map(|t| 10.0 + 2.0 * t as f64).collect();
        let m = train(&y, 1, 1, 0, 0.05, 800).unwrap();
        // one-step ahead: 10 + 2*30 = 70
        let f = forecast(&m, 1);
        assert!((f - 70.0).abs() < 0.01, "f={f}");
        // two-step ahead: 72
        let f2 = forecast(&m, 2);
        assert!((f2 - 72.0).abs() < 0.05, "f2={f2}");
    }

    #[test]
    fn ar1_series() {
        // y_t = 5 + 0.8·y_{t-1} + noise-free → forecast converges to 25
        let mut y = vec![5.0];
        for _ in 0..50 {
            let next = 5.0 + 0.8 * *y.last().unwrap();
            y.push(next);
        }
        let m = train(&y, 1, 0, 0, 0.01, 4000).unwrap();
        assert!((m.ar[0] - 0.8).abs() < 0.01, "ar={}", m.ar[0]);
        let f = forecast(&m, 1);
        let expect = 5.0 + 0.8 * y[50];
        assert!((f - expect).abs() < 0.1, "f={f} expect={expect}");
    }

    #[test]
    fn validation_errors() {
        assert!(train(&[1.0, 2.0, 3.0], 1, 1, 1, 0.05, 100).is_err());
        assert!(train(&[1.0; 20], 2, 0, 0, -0.05, 100).is_err());
        assert!(deserialize(&[0u8; 3]).is_err());
    }

    #[test]
    fn roundtrip() {
        let y: Vec<f64> = (0..20).map(|t| t as f64).collect();
        let m1 = train(&y, 1, 1, 0, 0.05, 500).unwrap();
        let m2 = deserialize(&serialize(&m1)).unwrap();
        assert_eq!(serialize(&m1), serialize(&m2));
        assert_eq!(forecast(&m2, 1), forecast(&m1, 1));
    }
}
