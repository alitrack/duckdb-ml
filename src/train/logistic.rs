use super::TrainingResult;
use std::error::Error;

/// Train logistic regression using gradient descent
pub fn train(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_epochs: usize,
) -> Result<TrainingResult, Box<dyn Error>> {
    let n_samples = x.len();
    let n_features = x[0].len();

    // Initialize weights (including intercept)
    let n_params = n_features + 1;
    let mut weights = vec![0.0; n_params];

    let mut prev_loss = f64::MAX;

    for _epoch in 0..max_epochs {
        let mut gradients = vec![0.0; n_params];
        let mut total_loss = 0.0_f64;

        for i in 0..n_samples {
            let mut z = weights[n_features];
            for j in 0..n_features {
                z += weights[j] * x[i][j];
            }

            let p = 1.0 / (1.0 + (-z).exp());
            let p = p.clamp(1e-15, 1.0 - 1e-15);

            total_loss += -(y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln());

            let error = p - y[i];
            for j in 0..n_features {
                #[allow(clippy::needless_range_loop)]
                {
                    gradients[j] += error * x[i][j];
                }
            }
            gradients[n_features] += error;
        }

        for g in gradients.iter_mut() {
            *g /= n_samples as f64;
        }
        total_loss /= n_samples as f64;

        for j in 0..n_params {
            weights[j] -= lr * gradients[j];
        }

        if (prev_loss - total_loss).abs() < 1e-6_f64 {
            log::info!("Logistic regression converged at epoch {_epoch}");
            break;
        }
        prev_loss = total_loss;
    }

    // Compute accuracy
    let mut correct = 0;
    for i in 0..n_samples {
        let mut z = weights[n_features];
        for j in 0..n_features {
            z += weights[j] * x[i][j];
        }
        let pred = if 1.0 / (1.0 + (-z).exp()) >= 0.5 {
            1.0
        } else {
            0.0
        };
        if (pred - y[i]).abs() < 1e-6_f64 {
            correct += 1;
        }
    }
    let _accuracy = Some(correct as f64 / n_samples as f64);

    Ok(TrainingResult {
        coefficients: weights.clone(),
        intercept: weights[n_features],
        r_squared: None,
        mse: None,
        num_samples: n_samples,
        model_blob: None,
    })
}
