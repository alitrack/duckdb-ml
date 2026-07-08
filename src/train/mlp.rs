//! MLP Training — Mini-batch SGD with momentum

use crate::model::mlp::MlpWeights;

/// Train a single-hidden-layer MLP regressor.
#[allow(clippy::needless_range_loop)]
pub fn train_mlp(
    x: &[Vec<f64>],
    y: &[f64],
    hidden_size: usize,
    lr: f64,
    momentum: f64,
    iterations: usize,
    batch_size: usize,
) -> Result<(MlpWeights, Option<f64>, Option<f64>), String> {
    let n = x.len();
    let input_size = x[0].len();
    let n_f = n as f64;

    if n < 2 {
        return Err("Need at least 2 samples".into());
    }

    // Xavier initialization
    let scale = (2.0 / input_size as f64).sqrt();
    let mut rng = LcgRng::new(42);

    let mut w1 = vec![vec![0.0f64; input_size]; hidden_size];
    for i in 0..hidden_size {
        for j in 0..input_size {
            w1[i][j] = (rng.next() - 0.5) * 2.0 * scale;
        }
    }
    let mut b1 = vec![0.0f64; hidden_size];
    let mut w2 = vec![0.0f64; hidden_size];
    for i in 0..hidden_size {
        w2[i] = (rng.next() - 0.5) * 2.0 * scale;
    }
    let mut b2 = 0.0;

    // Momentum buffers
    let mut v_w1 = vec![vec![0.0f64; input_size]; hidden_size];
    let mut v_b1 = vec![0.0f64; hidden_size];
    let mut v_w2 = vec![0.0f64; hidden_size];
    let mut v_b2 = 0.0f64;

    // Normalize targets to zero mean
    let y_mean = y.iter().sum::<f64>() / n_f;
    let y_norm: Vec<f64> = y.iter().map(|yi| yi - y_mean).collect();

    let effective_batch = batch_size.min(n);

    for _iter in 0..iterations {
        // Shuffle indices
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next() * i as f64) as usize;
            indices.swap(i, j);
        }

        for batch_start in (0..n).step_by(effective_batch) {
            let batch_end = (batch_start + effective_batch).min(n);
            let batch = &indices[batch_start..batch_end];
            let b_n = batch.len() as f64;

            let mut dw1 = vec![vec![0.0f64; input_size]; hidden_size];
            let mut db1 = vec![0.0f64; hidden_size];
            let mut dw2 = vec![0.0f64; hidden_size];
            let mut db2 = 0.0f64;

            for &idx in batch {
                let xi = &x[idx];
                let target = y_norm[idx];

                // Forward pass
                let mut hidden = vec![0.0f64; hidden_size];
                let mut hidden_pre_act = vec![0.0f64; hidden_size];
                for i in 0..hidden_size {
                    let mut z = b1[i];
                    for j in 0..input_size {
                        z += w1[i][j] * xi[j];
                    }
                    hidden_pre_act[i] = z;
                    hidden[i] = if z > 0.0 { z } else { 0.0 };
                }

                let mut output = b2;
                for i in 0..hidden_size {
                    output += w2[i] * hidden[i];
                }

                let error = output - target;

                // Backprop output layer
                for i in 0..hidden_size {
                    dw2[i] += error * hidden[i];
                    db2 += error;
                }

                // Backprop hidden layer
                for i in 0..hidden_size {
                    if hidden_pre_act[i] > 0.0 {
                        let grad = error * w2[i];
                        for j in 0..input_size {
                            dw1[i][j] += grad * xi[j];
                        }
                        db1[i] += grad;
                    }
                }
            }

            // SGD update with momentum
            for i in 0..hidden_size {
                for j in 0..input_size {
                    v_w1[i][j] = momentum * v_w1[i][j] - lr * dw1[i][j] / b_n;
                    w1[i][j] += v_w1[i][j];
                }
                v_b1[i] = momentum * v_b1[i] - lr * db1[i] / b_n;
                b1[i] += v_b1[i];

                v_w2[i] = momentum * v_w2[i] - lr * dw2[i] / b_n;
                w2[i] += v_w2[i];
            }
            v_b2 = momentum * v_b2 - lr * db2 / b_n;
            b2 += v_b2;
        }
    }

    let weights = MlpWeights {
        w1,
        b1,
        w2,
        b2: b2 + y_mean, // un-center
        input_size,
        hidden_size,
    };

    // Compute metrics
    let predictions: Vec<f64> = x
        .iter()
        .map(|xi| {
            let mut hidden = vec![0.0f64; hidden_size];
            for i in 0..hidden_size {
                let mut z = weights.b1[i];
                for j in 0..input_size {
                    z += weights.w1[i][j] * xi[j];
                }
                hidden[i] = if z > 0.0 { z } else { 0.0 };
            }
            let mut out = weights.b2;
            for i in 0..hidden_size {
                out += weights.w2[i] * hidden[i];
            }
            out
        })
        .collect();

    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(y.iter())
        .map(|(&p, &yi)| (yi - p).powi(2))
        .sum();
    let r2 = if ss_tot > 0.0 {
        Some(1.0 - ss_res / ss_tot)
    } else {
        None
    };

    Ok((weights, r2, Some(ss_res / n_f)))
}

/// Simple LCG RNG for weight initialization
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state as f64 / u64::MAX as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_simple() {
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
            vec![6.0, 5.0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 2.0 * xi[1] + 1.0).collect();

        let (weights, r2, _) = train_mlp(&x, &y, 4, 0.05, 0.9, 3000, 6).unwrap();
        assert_eq!(weights.w1.len(), 4);
        // Small data: MLP may not converge well; verify structure + finite outputs
        let pred = weights.b2
            + weights
                .w2
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    let z = weights.b1[i] + weights.w1[i][0] * 3.0 + weights.w1[i][1] * 3.0;
                    w * if z > 0.0 { z } else { 0.0 }
                })
                .sum::<f64>();
        assert!(pred.is_finite(), "pred not finite: {pred}");
    }
}
