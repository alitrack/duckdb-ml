pub mod linear;
pub mod logistic;
pub mod tree;

use crate::model::Algorithm;

use std::error::Error;

/// Training result
pub struct TrainingResult {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub r_squared: Option<f64>,
    pub mse: Option<f64>,
    pub num_samples: usize,
    /// Serialized model data for non-linear models (decision tree, random forest)
    pub model_blob: Option<Vec<u8>>,
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
        Algorithm::DecisionTreeRegressor => {
            let max_depth = params.get("max_depth").copied().unwrap_or(10.0) as usize;
            let min_samples_split =
                params.get("min_samples_split").copied().unwrap_or(5.0) as usize;
            let tp = tree::TreeParams {
                max_depth,
                min_samples_split,
                min_samples_leaf: params.get("min_samples_leaf").copied().unwrap_or(2.0) as usize,
                max_features: None,
            };
            let tree_node = tree::build_tree(x, y, &tp);
            let blob = tree_node.to_bytes();
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::RandomForestRegressor => {
            let n_estimators = params.get("n_estimators").copied().unwrap_or(100.0) as usize;
            let max_depth = params.get("max_depth").copied().unwrap_or(10.0) as usize;
            let tp = tree::TreeParams {
                max_depth,
                min_samples_split: params.get("min_samples_split").copied().unwrap_or(5.0) as usize,
                min_samples_leaf: params.get("min_samples_leaf").copied().unwrap_or(2.0) as usize,
                max_features: None,
            };
            let forest = tree::RandomForest::train(x, y, n_estimators, &tp);
            // Serialize forest
            let mut buf = Vec::new();
            let count = forest.trees.len() as u32;
            buf.extend_from_slice(&count.to_le_bytes());
            for t in &forest.trees {
                let tb = t.to_bytes();
                let len = tb.len() as u32;
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&tb);
            }
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: Some(buf),
            })
        }
        Algorithm::Onnx => Err(
            "ONNX models cannot be trained in DuckDB. Train in Python and load via ml_load_onnx()."
                .into(),
        ),
    }
}
