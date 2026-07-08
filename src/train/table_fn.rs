//! ml_train table function — end-to-end: SQL data → train model → store → register → predict
//!
//! SQL: SELECT * FROM ml_train('name', 'algo', 'target_json', 'features_json', 'params_json')
//!
//! Parameters (all VARCHAR):
//!   0: model_name
//!   1: algorithm
//!   2: target  — JSON array of f64, e.g. "[10.0, 20.0, 30.0]"
//!   3: features — JSON array of arrays, e.g. "[[1.0,2.0],[3.0,4.0],[5.0,6.0]]"
//!   4: params_json (optional) — JSON object, e.g. '{"lr": 0.01}'

use crate::model::{Algorithm, global_registry};
use crate::train;
use duckdb::{
    Result,
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[repr(C)]
pub struct TInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct TBindData {
    model_name: String,
    algorithm: String,
    r_squared: Option<f64>,
    mse: Option<f64>,
    n_samples: usize,
    n_features: usize,
}

pub struct TrainFn;

impl VTab for TrainFn {
    type BindData = TBindData;
    type InitData = TInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 4 {
            return Err("ml_train requires: model_name, algorithm, target_json, features_json [, params_json]".into());
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let algorithm_str: String = bind.get_parameter(1).to_string();
        let target_json: String = bind.get_parameter(2).to_string();
        let features_json: String = bind.get_parameter(3).to_string();
        let params_json: String = if n_params >= 5 {
            bind.get_parameter(4).to_string()
        } else {
            "{}".into()
        };

        let algorithm = Algorithm::parse_algorithm(&algorithm_str)
            .ok_or_else(|| format!("Unknown algorithm: '{algorithm_str}'. Available: linear_regression, ridge_regression, logistic_regression, decision_tree, random_forest, kmeans, knn_regressor, knn_classifier, naive_bayes, pca"))?;

        // Parse target array
        let y: Vec<f64> =
            serde_json::from_str(&target_json).map_err(|e| format!("Invalid target JSON: {e}"))?;

        // Parse features matrix
        let x: Vec<Vec<f64>> = serde_json::from_str(&features_json)
            .map_err(|e| format!("Invalid features JSON: {e}"))?;

        if x.is_empty() || y.is_empty() {
            return Err("Training data is empty".into());
        }
        if x.len() != y.len() {
            return Err(format!(
                "Sample count mismatch: {} features rows vs {} target values",
                x.len(),
                y.len()
            )
            .into());
        }
        let n_features = x[0].len();
        if !x.iter().all(|row| row.len() == n_features) {
            return Err("Features rows have inconsistent column counts".into());
        }

        // Parse params
        let params: HashMap<String, f64> = if params_json.trim().is_empty() || params_json == "{}" {
            HashMap::new()
        } else {
            serde_json::from_str(&params_json).map_err(|e| format!("Invalid params JSON: {e}"))?
        };

        let n_samples = x.len();
        let n_feat = n_features;

        let result = train::train(algorithm, &x, &y, &params)?;

        // Build hyperparameters JSON
        let mut hp = String::from("{");
        let mut first = true;
        for (k, v) in &params {
            if !first {
                hp.push_str(", ");
            }
            let _ = write!(hp, "\"{k}\": {v}");
            first = false;
        }
        hp.push('}');

        // Serialize and store model blob → load as MlModel → register
        if let Some(ref blob) = result.model_blob {
            // Create MlModel from blob and register
            let registered = register_from_blob(algorithm, n_feat, n_samples, &params, blob);
            if let Ok(arc_model) = registered {
                global_registry().insert(model_name.clone(), arc_model);
            }
        } else {
            // Linear/logistic/lasso models: construct MlModel directly
            use crate::model::{lasso::LassoModel, linear::LinearModel, logistic::LogisticModel};
            let lambda = params.get("lambda").copied().unwrap_or(0.0);
            let model: Arc<dyn crate::model::MlModel> = match algorithm {
                Algorithm::LinearRegression | Algorithm::RidgeRegression => {
                    Arc::new(LinearModel::new(
                        result.coefficients,
                        n_samples,
                        result.r_squared,
                        result.mse,
                        lambda,
                    ))
                }
                Algorithm::LogisticRegression => Arc::new(LogisticModel::new(
                    result.coefficients,
                    n_samples,
                    result.r_squared,
                )),
                Algorithm::LassoRegression => {
                    let l = params.get("lambda").copied().unwrap_or(0.1);
                    Arc::new(LassoModel::new(
                        result.coefficients,
                        n_samples,
                        result.r_squared,
                        result.mse,
                        l,
                    ))
                }
                _ => {
                    return Err("Linear training result for non-linear algorithm".into());
                }
            };
            global_registry().insert(model_name.clone(), model);
        }

        // Cache features for batch prediction via @model_name syntax
        global_registry().cache_dataset(&model_name, x.clone());

        bind.add_result_column(
            "model_name",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column("algorithm", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("r_squared", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("mse", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("n_samples", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column(
            "n_features",
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        );

        Ok(TBindData {
            model_name,
            algorithm: algorithm_str,
            r_squared: result.r_squared,
            mse: result.mse,
            n_samples,
            n_features: n_feat,
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(TInitData {
            done: AtomicBool::new(false),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        let bind = func.get_bind_data();

        use arrow::array::{Float64Array, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(Schema::new(vec![
            Field::new("model_name", DataType::Utf8, false),
            Field::new("algorithm", DataType::Utf8, false),
            Field::new("r_squared", DataType::Float64, true),
            Field::new("mse", DataType::Float64, true),
            Field::new("n_samples", DataType::Int32, false),
            Field::new("n_features", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![bind.model_name.as_str()])),
                Arc::new(StringArray::from(vec![bind.algorithm.as_str()])),
                Arc::new(Float64Array::from(vec![bind.r_squared])),
                Arc::new(Float64Array::from(vec![bind.mse])),
                Arc::new(Int32Array::from(vec![Some(bind.n_samples as i32)])),
                Arc::new(Int32Array::from(vec![Some(bind.n_features as i32)])),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        init.done.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        None
    }
}

/// Deserialize a blob into an MlModel and return as Arc
fn register_from_blob(
    algorithm: Algorithm,
    _n_features: usize,
    _n_samples: usize,
    _params: &HashMap<String, f64>,
    blob: &[u8],
) -> Result<Arc<dyn crate::model::MlModel>, Box<dyn Error>> {
    use crate::model::{
        MlModel,
        kmeans::KMeansModel,
        knn::KnnMlModel,
        linear::LinearModel,
        logistic::LogisticModel,
        naive_bayes::NbMlModel,
        pca::PcaMlModel,
        tree::{ForestModel, TreeModel},
    };

    #[cfg(feature = "onnx")]
    use crate::model::onnx::OnnxModel;

    #[cfg(feature = "onnx")]
    {
        if matches!(algorithm, Algorithm::Onnx) {
            let model = OnnxModel::deserialize(blob)?;
            return Ok(Arc::new(model));
        }
    }

    let model: Arc<dyn MlModel> = match algorithm {
        Algorithm::LinearRegression | Algorithm::RidgeRegression => {
            Arc::new(LinearModel::deserialize(blob)?)
        }
        Algorithm::LogisticRegression => Arc::new(LogisticModel::deserialize(blob)?),
        Algorithm::DecisionTreeRegressor => Arc::new(TreeModel::deserialize(blob)?),
        Algorithm::RandomForestRegressor => Arc::new(ForestModel::deserialize(blob)?),
        Algorithm::KMeans => Arc::new(KMeansModel::deserialize(blob)?),
        Algorithm::KNNRegressor | Algorithm::KNNClassifier => {
            Arc::new(KnnMlModel::deserialize(blob)?)
        }
        Algorithm::XGBoostRegression | Algorithm::XGBoostBinary => {
            // GBDT models are serialized as XGBoost JSON
            use crate::model::xgboost::XgbModelWrapper;
            let model = XgbModelWrapper::new(blob.to_vec())
                .map_err(|e| format!("GBDT deserialize: {e:?}"))?;
            return Ok(Arc::new(model));
        }
        Algorithm::NaiveBayes => Arc::new(NbMlModel::deserialize(blob)?),
        Algorithm::PCA => Arc::new(PcaMlModel::deserialize(blob)?),
        Algorithm::XGBoostRegressor
        | Algorithm::XGBoostClassifier
        | Algorithm::Onnx
        | Algorithm::LassoRegression => {
            return Err(
                "XGBoost and ONNX models must be loaded via ml_load_onnx/ml_load_xgboost".into(),
            );
        }
    };
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify train + register_from_blob produces a working MlModel
    fn roundtrip(algo: Algorithm, x: &[Vec<f64>], y: &[f64], params: &HashMap<String, f64>) {
        let result = crate::train::train(algo, x, y, params).unwrap();

        let blob = if let Some(ref b) = result.model_blob {
            b.clone()
        } else {
            result
                .coefficients
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .chain(result.intercept.to_le_bytes())
                .collect()
        };

        let model = register_from_blob(algo, x[0].len(), x.len(), params, &blob).unwrap();
        let pred = model.predict(&x[0]).unwrap();
        assert!(pred.is_finite(), "prediction should be finite: {pred}");
    }

    #[test]
    fn test_ml_train_linear() {
        // Linear models use direct construction (bincode, not blob). Test via registry.
        use crate::model::linear::LinearModel;
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
        ];
        let y = vec![8.0, 7.0, 18.0, 17.0];
        let result =
            crate::train::train(Algorithm::LinearRegression, &x, &y, &HashMap::new()).unwrap();
        let model = LinearModel::new(result.coefficients, 4, result.r_squared, result.mse, 0.0);
        global_registry().insert("test_linear".into(), Arc::new(model));
        let m = global_registry().get("test_linear").unwrap();
        let pred = m.predict(&[3.0, 4.0]).unwrap();
        assert!(pred.is_finite(), "pred={pred}");
        assert!((pred - 18.0).abs() < 5.0, "linear pred={pred}");
    }

    #[test]
    fn test_ml_train_knn() {
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let mut params = HashMap::new();
        params.insert("k".into(), 1.0);
        roundtrip(Algorithm::KNNRegressor, &x, &y, &params);
    }

    #[test]
    fn test_ml_train_kmeans() {
        let x = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut params = HashMap::new();
        params.insert("k".into(), 2.0);
        roundtrip(Algorithm::KMeans, &x, &y, &params);
    }

    #[test]
    fn test_ml_train_naive_bayes() {
        let x = vec![
            vec![1.0, 1.0],
            vec![1.2, 1.2],
            vec![5.0, 5.0],
            vec![4.8, 4.8],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        roundtrip(Algorithm::NaiveBayes, &x, &y, &HashMap::new());
    }

    #[test]
    fn test_ml_train_pca() {
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0];
        let mut params = HashMap::new();
        params.insert("k".into(), 2.0);
        roundtrip(Algorithm::PCA, &x, &y, &params);
    }
}
