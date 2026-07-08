pub mod gbdt;
pub mod kmeans;
pub mod linear;
pub mod logistic;
pub mod table_fn;
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
        Algorithm::XGBoostRegression | Algorithm::XGBoostBinary => {
            let n_estimators = params.get("n_estimators").copied().unwrap_or(100.0) as usize;
            let learning_rate = params.get("learning_rate").copied().unwrap_or(0.1);
            let max_depth = params.get("max_depth").copied().unwrap_or(6.0) as usize;
            let subsample = params.get("subsample").copied().unwrap_or(1.0);
            let gp = gbdt::GbdtParams {
                n_estimators,
                learning_rate,
                max_depth,
                subsample,
                ..Default::default()
            };
            let ensemble = gbdt::train_gbdt(x, y, &gp);
            let r2 = ensemble.r_squared(x, y);
            let mse_val = ensemble.mse(x, y);
            let json = ensemble.to_xgb_json();
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: ensemble.initial_prediction,
                r_squared: Some(r2),
                mse: Some(mse_val),
                num_samples: x.len(),
                model_blob: Some(json.into_bytes()),
            })
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
            let mut blob = Vec::new();
            blob.extend_from_slice(&(x[0].len() as u32).to_le_bytes());
            blob.extend_from_slice(&tree_node.to_bytes());
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
            // Header: num_features (4 bytes)
            buf.extend_from_slice(&(x[0].len() as u32).to_le_bytes());
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
        Algorithm::XGBoostRegressor | Algorithm::XGBoostClassifier => {
            // v0.9 compat: external model loading via ml_load_xgboost
            // For training, use xgboost_regression or xgboost_binary
            Err(
                "XGBoost models trained outside DuckDB. Use 'xgboost_regression' or 'xgboost_binary' for in-DB training."
                    .into(),
            )
        }
        Algorithm::KMeans => {
            let k = params.get("k").copied().unwrap_or(3.0) as usize;
            let max_iters = params.get("max_iters").copied().unwrap_or(100.0) as usize;
            let tol = params.get("tol").copied().unwrap_or(1e-4);
            let result = kmeans::train(x, k, max_iters, tol);
            let blob = kmeans::serialize_centroids(&result.centroids);
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: result.labels.is_empty().then_some(0.0), // inertia stored as metadata
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::KNNRegressor => {
            let k = params.get("k").copied().unwrap_or(5.0) as usize;
            let model = crate::model::knn::KnnModel::new(
                x.to_vec(),
                y.to_vec(),
                k,
                crate::model::knn::KnnTask::Regression,
            );
            let blob = model.to_bytes();
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::KNNClassifier => {
            let k = params.get("k").copied().unwrap_or(5.0) as usize;
            let model = crate::model::knn::KnnModel::new(
                x.to_vec(),
                y.to_vec(),
                k,
                crate::model::knn::KnnTask::Classification,
            );
            let blob = model.to_bytes();
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::NaiveBayes => {
            let model = crate::model::naive_bayes::NaiveBayesModel::train(x, y);
            let blob = model.to_bytes();
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::PCA => {
            let k = params.get("k").copied().unwrap_or(2.0) as usize;
            let model = crate::model::pca::PcaModel::fit(x, k);
            let blob = model.to_bytes();
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
    }
}
