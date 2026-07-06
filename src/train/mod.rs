pub mod linear;
pub mod logistic;

use crate::model::Algorithm;

use std::error::Error;

/// Training result
pub struct TrainingResult {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub r_squared: Option<f64>,
    pub mse: Option<f64>,
    pub num_samples: usize,
}

/// Train a model given a feature matrix (n_samples × n_features) and target vector
pub fn train(
    algorithm: Algorithm,
    x: &[Vec<f64>],
    y: &[f64],
    params: &std::collections::HashMap<String, f64>,
) -> Result<TrainingResult, Box<dyn Error>> {
    match algorithm {
        Algorithm::LinearRegression | Algorithm::RidgeRegression => {
            let lambda = params.get("lambda").copied().unwrap_or(0.0);
            linear::train(x, y, lambda)
        }
        Algorithm::LogisticRegression => {
            let lr = params.get("lr").copied().unwrap_or(0.01);
            let epochs = params.get("epochs").copied().unwrap_or(100.0) as usize;
            logistic::train(x, y, lr, epochs)
        }
    }
}
