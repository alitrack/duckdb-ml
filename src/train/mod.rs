pub mod adaboost;
pub mod agglomerative;
pub mod arima;
pub mod cox;
pub mod dbscan;
pub mod elastic_net;
pub mod fcm;
pub mod gbdt;
pub mod km;
pub mod kmeans;
pub mod lasso;
pub mod linear;
pub mod logistic;
pub mod mlp;
pub mod multilogistic;
pub mod ordinal;
pub mod poly;
pub mod robust;
pub mod smote;
pub mod svm;
pub mod svr;
pub mod table_fn;
pub mod tree;

/// Serialize a forest (tree list) to the model blob format:
/// num_features u32 · tree_count u32 · per-tree [len u32, tree bytes]
fn forest_model_to_blob(
    trees: &[tree::TreeNode],
    n_features: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(n_features as u32).to_le_bytes());
    let count = trees.len() as u32;
    buf.extend_from_slice(&count.to_le_bytes());
    for t in trees {
        let tb = t.to_bytes();
        let len = tb.len() as u32;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&tb);
    }
    Ok(buf)
}

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
        Algorithm::PolynomialRegression => {
            let degree = params.get("degree").copied().unwrap_or(2.0) as usize;
            let lambda = params.get("lambda").copied().unwrap_or(0.0);
            poly::train(x, y, degree, lambda)
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
            // multi-class: xgboost_binary + num_class > 2 → multi:softprob
            let num_class = params.get("num_class").copied().unwrap_or(0.0) as usize;
            let (ensemble, objective) = if algorithm == Algorithm::XGBoostBinary && num_class > 2 {
                let e = gbdt::train_gbdt_softmax(x, y, &gp, num_class);
                (e, "multi:softprob")
            } else {
                let gbdt_objective = match algorithm {
                    Algorithm::XGBoostBinary => gbdt::GbdtObjective::Logistic,
                    _ => gbdt::GbdtObjective::SquaredError,
                };
                let e = gbdt::train_gbdt(x, y, &gp, gbdt_objective);
                let obj = match gbdt_objective {
                    gbdt::GbdtObjective::Logistic => "binary:logistic",
                    gbdt::GbdtObjective::SquaredError => "reg:squarederror",
                    gbdt::GbdtObjective::Softmax { .. } => unreachable!(),
                };
                (e, obj)
            };
            let r2 = ensemble.r_squared(x, y);
            let mse_val = ensemble.mse(x, y);
            let json = ensemble.to_xgb_json(objective);
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
                min_samples_split: params.get("min_samples_split").copied().unwrap_or(2.0) as usize,
                min_samples_leaf: params.get("min_samples_leaf").copied().unwrap_or(1.0) as usize,
                max_features: None,
            };
            let forest = tree::RandomForest::train(x, y, n_estimators, &tp);
            let blob = forest_model_to_blob(&forest.trees, x[0].len())?;
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: y.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::RandomForestClassifier => {
            let n_estimators = params.get("n_estimators").copied().unwrap_or(100.0) as usize;
            let max_depth = params.get("max_depth").copied().unwrap_or(10.0) as usize;
            let tp = tree::TreeParams {
                max_depth,
                min_samples_split: params.get("min_samples_split").copied().unwrap_or(2.0) as usize,
                min_samples_leaf: params.get("min_samples_leaf").copied().unwrap_or(1.0) as usize,
                max_features: None,
            };
            let forest = tree::RandomForestClassifier::train(x, y, n_estimators, &tp);
            let blob = forest_model_to_blob(&forest.trees, x[0].len())?;
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: y.len(),
                model_blob: Some(blob),
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
        Algorithm::DBSCAN => {
            let eps = params.get("eps").copied().unwrap_or(0.5);
            let min_points = params.get("min_points").copied().unwrap_or(5.0) as usize;
            let result = dbscan::train(x, eps, min_points)
                .map_err(|e| -> Box<dyn Error> { format!("dbscan: {e}").into() })?;
            let blob = dbscan::serialize(&result.clusters, x[0].len(), eps);
            let noise_ratio = if x.is_empty() {
                0.0
            } else {
                result.noise_count as f64 / x.len() as f64
            };
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: Some(noise_ratio),
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::MultinomialLogisticRegression => {
            let lr = params.get("lr").copied().unwrap_or(0.1);
            let max_epochs = params.get("max_epochs").copied().unwrap_or(500.0) as usize;
            let m = multilogistic::train(x, y, lr, max_epochs)
                .map_err(|e| -> Box<dyn Error> { format!("multilogistic: {e}").into() })?;
            let blob = multilogistic::serialize(&m);
            // accuracy on training data
            let mut correct = 0;
            for i in 0..x.len() {
                if multilogistic::predict_one(&m, &x[i]) == y[i] {
                    correct += 1;
                }
            }
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: Some(correct as f64 / x.len() as f64),
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::OrdinalLogisticRegression => {
            let lr = params.get("lr").copied().unwrap_or(0.1);
            let max_epochs = params.get("max_epochs").copied().unwrap_or(800.0) as usize;
            let m = ordinal::train(x, y, lr, max_epochs)
                .map_err(|e| -> Box<dyn Error> { format!("ordinal: {e}").into() })?;
            let blob = ordinal::serialize(&m);
            let mut correct = 0;
            for i in 0..x.len() {
                if ordinal::predict_one(&m, &x[i]) == y[i] {
                    correct += 1;
                }
            }
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: Some(correct as f64 / x.len() as f64),
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::CoxProportionalHazards => {
            Err("cox models are trained via ml_cox_train(model, time, event, features, params)".into())
        }
        Algorithm::KaplanMeier => {
            Err("kaplan-meier models are trained via ml_km_train(model, time, event)".into())
        }
        Algorithm::RobustRegression => {
            let c = params.get("c").copied().unwrap_or(1.345);
            let max_iters = params.get("max_iters").copied().unwrap_or(50.0) as usize;
            robust::train(x, y, c, max_iters)
                .map_err(|e| -> Box<dyn Error> { format!("robust: {e}").into() })
        }
        Algorithm::ElasticNetRegression => {
            let alpha = params.get("alpha").copied().unwrap_or(1.0);
            let l1_ratio = params.get("l1_ratio").copied().unwrap_or(0.5);
            let max_iter = params.get("max_iter").copied().unwrap_or(1000.0) as usize;
            elastic_net::train(x, y, alpha, l1_ratio, max_iter)
                .map_err(|e| -> Box<dyn Error> { format!("elastic_net: {e}").into() })
        }
        Algorithm::Arima => {
            let p = params.get("p").copied().unwrap_or(1.0) as usize;
            let d = params.get("d").copied().unwrap_or(1.0) as usize;
            let q = params.get("q").copied().unwrap_or(0.0) as usize;
            let lr = params.get("lr").copied().unwrap_or(0.05);
            let max_epochs = params.get("max_epochs").copied().unwrap_or(1000.0) as usize;
            let m = arima::train(y, p, d, q, lr, max_epochs)
                .map_err(|e| -> Box<dyn Error> { format!("arima: {e}").into() })?;
            let blob = arima::serialize(&m);
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: m.intercept,
                r_squared: None,
                mse: None,
                num_samples: y.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::SVR => {
            let kernel = match params.get("kernel").copied().unwrap_or(1.0) as u8 {
                0 => svr::Kernel::Linear,
                1 => svr::Kernel::Rbf,
                2 => svr::Kernel::Polynomial,
                3 => svr::Kernel::Sigmoid,
                _ => return Err("svr: unknown kernel code (0=linear|1=rbf|2=poly|3=sigmoid)".into()),
            };
            let c = params.get("c").copied().unwrap_or(1.0);
            let epsilon = params.get("epsilon").copied().unwrap_or(0.1);
            let gamma = params.get("gamma").copied().unwrap_or(0.0);
            let degree = params.get("degree").copied().unwrap_or(3.0) as usize;
            let coef0 = params.get("coef0").copied().unwrap_or(0.0);
            let tol = params.get("tol").copied().unwrap_or(1e-3);
            let max_iter = params.get("max_iter").copied().unwrap_or(2000.0) as usize;
            svr::train(x, y, kernel, c, epsilon, gamma, degree, coef0, tol, max_iter)
                .map_err(|e| -> Box<dyn Error> { format!("svr: {e}").into() })
        }
        Algorithm::SVM => {
            let c = params.get("c").copied().unwrap_or(1.0);
            let kernel = params.get("kernel").copied().unwrap_or(1.0) as u8;
            let gamma = params.get("gamma").copied().unwrap_or(1.0);
            let degree = params.get("degree").copied().unwrap_or(3.0);
            let coef0 = params.get("coef0").copied().unwrap_or(0.0);
            let t = svm::train(x, y, c, kernel, gamma, degree, coef0)
                .map_err(|e| -> Box<dyn Error> { format!("svm: {e}").into() })?;
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: t.rho,
                r_squared: None,
                mse: Some(t.n_support as f64),
                num_samples: x.len(),
                model_blob: Some(t.blob),
            })
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
        Algorithm::FuzzyCMeans => {
            let k = params.get("k").copied().unwrap_or(3.0) as usize;
            let m = params.get("fuzziness").copied().unwrap_or(2.0);
            let max_iters = params.get("max_iters").copied().unwrap_or(100.0) as usize;
            let tol = params.get("tol").copied().unwrap_or(1e-4);
            let result = fcm::train(x, k, m, max_iters, tol);
            // centroids flow through TrainingResult.coefficients; FcmModel is
            // constructed in table_fn (keeps fuzziness + iterations in metadata)
            let flat: Vec<f64> = result.centroids.iter().flatten().copied().collect();
            Ok(TrainingResult {
                coefficients: flat,
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: None,
            })
        }
        Algorithm::Agglomerative => {
            let k = params.get("k").copied().unwrap_or(3.0) as usize;
            let linkage_code = params.get("linkage").copied().unwrap_or(1.0) as usize;
            let lnk = match linkage_code {
                0 => agglomerative::Linkage::Single,
                1 => agglomerative::Linkage::Complete,
                _ => agglomerative::Linkage::Average,
            };
            let result = agglomerative::train(x, k, lnk);
            let m = crate::model::agglomerative::AgglomerativeModel::new(
                &result,
                x[0].len(),
                x.len(),
                lnk,
            );
            let blob = crate::model::MlModel::serialize(&m)
                .map_err(|e| -> Box<dyn Error> { format!("agglomerative serialize: {e}").into() })?;
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
                num_samples: x.len(),
                model_blob: Some(blob),
            })
        }
        Algorithm::AdaBoost => {
            let n_estimators = params.get("n_estimators").copied().unwrap_or(50.0) as usize;
            let result = adaboost::train(x, y, n_estimators);
            let n_feats = if x.is_empty() { 0 } else { x[0].len() };
            let m =
                crate::model::adaboost::AdaBoostModel::new(&result, n_feats, x.len());
            let blob = crate::model::MlModel::serialize(&m)
                .map_err(|e| -> Box<dyn Error> { format!("adaboost serialize: {e}").into() })?;
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared: None,
                mse: None,
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
        Algorithm::LassoRegression => {
            let lambda = params.get("lambda").copied().unwrap_or(0.1);
            let max_iter = params.get("max_iter").copied().unwrap_or(1000.0) as usize;
            let tol = params.get("tol").copied().unwrap_or(1e-4);
            let (coef, r_squared, mse) = lasso::train_lasso(x, y, lambda, max_iter, tol)?;
            Ok(TrainingResult {
                coefficients: coef.clone(),
                intercept: *coef.last().unwrap_or(&0.0),
                r_squared,
                mse,
                num_samples: x.len(),
                model_blob: None, // stored via coefficients for Linear-style models
            })
        }
        Algorithm::MlpRegressor => {
            let hidden_size = params.get("hidden_size").copied().unwrap_or(8.0) as usize;
            let lr = params.get("lr").copied().unwrap_or(0.01);
            let momentum = params.get("momentum").copied().unwrap_or(0.9);
            let iterations = params.get("iterations").copied().unwrap_or(200.0) as usize;
            let batch_size = params.get("batch_size").copied().unwrap_or(16.0) as usize;
            let (weights, r_squared, mse) =
                mlp::train_mlp(x, y, hidden_size, lr, momentum, iterations, batch_size)?;
            let blob = bincode::encode_to_vec(&weights, bincode::config::standard())
                .map_err(|e| e.to_string())?;
            Ok(TrainingResult {
                coefficients: vec![],
                intercept: 0.0,
                r_squared,
                mse,
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
        Algorithm::LDA => {
            let k = params.get("k").copied().unwrap_or(2.0) as usize;
            let model = crate::model::lda::LdaModel::fit(x, y, k)
                .ok_or("lda: need >= 2 classes and a non-singular within-class scatter (try more samples per class)")?;
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
