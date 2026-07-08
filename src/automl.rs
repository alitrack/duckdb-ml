//! ml_compare table function — AutoML: train multiple algorithms and compare
//!
//! SQL: SELECT * FROM ml_compare('exp_name', 'target_json', 'features_json',
//!                               'algorithms_json', 'task_type', 'params_json')
//!
//! Trains all specified (or auto-selected) algorithms, registers them in the
//! global registry, and returns a comparison table.

use crate::model::{Algorithm, global_registry};
use crate::train;
use duckdb::{
    Result,
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Mutex;

/// Shuttle: bind() stores results here, func() reads and clears them.
/// DuckDB vtab API calls bind() before init()/func() in the same thread,
/// so a thread-local static is safe.
static COMPARE_RESULTS: std::sync::LazyLock<Mutex<Vec<CompareRow>>> =
    std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Clone)]
struct CompareRow {
    model_name: String,
    algorithm: String,
    r_squared: Option<f64>,
    mse: Option<f64>,
    n_samples: i32,
    n_features: i32,
}

#[repr(C)]
pub struct CInitData {
    rows: Vec<CompareRow>,
    cursor: Mutex<usize>,
}

pub struct CompareFn;

impl VTab for CompareFn {
    type BindData = ();
    type InitData = CInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 4 {
            return Err("ml_compare requires: experiment_name, target_json, features_json, algorithms_json [, task_type, params_json]".into());
        }

        let exp_name: String = bind.get_parameter(0).to_string();
        let target_json: String = bind.get_parameter(1).to_string();
        let features_json: String = bind.get_parameter(2).to_string();
        let algorithms_json: String = bind.get_parameter(3).to_string();
        let task_type: String = if n_params >= 5 {
            bind.get_parameter(4).to_string()
        } else {
            "regression".into()
        };
        let params_json: String = if n_params >= 6 {
            bind.get_parameter(5).to_string()
        } else {
            "{}".into()
        };

        let y: Vec<f64> =
            serde_json::from_str(&target_json).map_err(|e| format!("Invalid target JSON: {e}"))?;
        let x: Vec<Vec<f64>> = serde_json::from_str(&features_json)
            .map_err(|e| format!("Invalid features JSON: {e}"))?;
        let algo_names: Vec<String> = serde_json::from_str(&algorithms_json)
            .map_err(|e| format!("Invalid algorithms JSON: {e}"))?;
        let params: HashMap<String, f64> = if params_json.trim().is_empty() || params_json == "{}" {
            HashMap::new()
        } else {
            serde_json::from_str(&params_json).map_err(|e| format!("Invalid params JSON: {e}"))?
        };

        let algo_names = if algo_names.is_empty() {
            match task_type.as_str() {
                "classification" => vec![
                    "logistic_regression".into(),
                    "random_forest".into(),
                    "knn_classifier".into(),
                    "naive_bayes".into(),
                ],
                _ => vec![
                    "linear_regression".into(),
                    "random_forest".into(),
                    "knn_regressor".into(),
                ],
            }
        } else {
            algo_names
        };

        let mut results = Vec::new();
        let registry = global_registry();

        for algo_name in &algo_names {
            let algorithm = match Algorithm::parse_algorithm(algo_name) {
                Some(a) => a,
                None => continue,
            };

            let model_key = format!("{exp_name}_{algo_name}");

            match train::train(algorithm, &x, &y, &params) {
                Ok(result) => {
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
                    use std::sync::Arc;

                    let arc_model: Option<Arc<dyn MlModel>> =
                        if let Some(ref blob) = result.model_blob {
                            match algorithm {
                                Algorithm::DecisionTreeRegressor => TreeModel::deserialize(blob)
                                    .ok()
                                    .map(|m| Arc::new(m) as Arc<dyn MlModel>),
                                Algorithm::RandomForestRegressor => ForestModel::deserialize(blob)
                                    .ok()
                                    .map(|m| Arc::new(m) as Arc<dyn MlModel>),
                                Algorithm::KMeans => KMeansModel::deserialize(blob)
                                    .ok()
                                    .map(|m| Arc::new(m) as Arc<dyn MlModel>),
                                Algorithm::KNNRegressor | Algorithm::KNNClassifier => {
                                    KnnMlModel::deserialize(blob)
                                        .ok()
                                        .map(|m| Arc::new(m) as Arc<dyn MlModel>)
                                }
                                Algorithm::NaiveBayes => NbMlModel::deserialize(blob)
                                    .ok()
                                    .map(|m| Arc::new(m) as Arc<dyn MlModel>),
                                Algorithm::PCA => PcaMlModel::deserialize(blob)
                                    .ok()
                                    .map(|m| Arc::new(m) as Arc<dyn MlModel>),
                                Algorithm::XGBoostRegression | Algorithm::XGBoostBinary => {
                                    crate::model::xgboost::XgbModelWrapper::new(blob.to_vec())
                                        .ok()
                                        .map(|m| Arc::new(m) as Arc<dyn MlModel>)
                                }
                                _ => None,
                            }
                        } else {
                            match algorithm {
                                Algorithm::LinearRegression | Algorithm::RidgeRegression => {
                                    let lambda = params.get("lambda").copied().unwrap_or(0.0);
                                    Some(Arc::new(LinearModel::new(
                                        result.coefficients,
                                        result.num_samples,
                                        result.r_squared,
                                        result.mse,
                                        lambda,
                                    )) as Arc<dyn MlModel>)
                                }
                                Algorithm::LogisticRegression => Some(Arc::new(LogisticModel::new(
                                    result.coefficients,
                                    result.num_samples,
                                    result.r_squared,
                                ))
                                    as Arc<dyn MlModel>),
                                _ => None,
                            }
                        };

                    if let Some(m) = arc_model {
                        registry.insert(model_key.clone(), m);
                    }

                    results.push(CompareRow {
                        model_name: model_key,
                        algorithm: algo_name.clone(),
                        r_squared: result.r_squared,
                        mse: result.mse,
                        n_samples: x.len() as i32,
                        n_features: x[0].len() as i32,
                    });
                }
                Err(e) => {
                    results.push(CompareRow {
                        model_name: model_key,
                        algorithm: format!("{algo_name} (ERR: {e})"),
                        r_squared: None,
                        mse: None,
                        n_samples: 0,
                        n_features: 0,
                    });
                }
            }
        }

        // Store results through the shuttle
        *COMPARE_RESULTS.lock().unwrap() = results;

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

        Ok(())
    }

    fn init(_init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        let rows = COMPARE_RESULTS.lock().unwrap().clone();
        Ok(CInitData {
            rows,
            cursor: Mutex::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        let mut cursor = init.cursor.lock().unwrap();
        let remaining = init.rows.len().saturating_sub(*cursor);
        let chunk_size = 2048.min(remaining);

        if chunk_size == 0 {
            output.set_len(0);
            return Ok(());
        }

        use arrow::array::{Float64Array, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("model_name", DataType::Utf8, false),
            Field::new("algorithm", DataType::Utf8, false),
            Field::new("r_squared", DataType::Float64, true),
            Field::new("mse", DataType::Float64, true),
            Field::new("n_samples", DataType::Int32, false),
            Field::new("n_features", DataType::Int32, false),
        ]));

        let slice = &init.rows[*cursor..*cursor + chunk_size];
        let names: Vec<&str> = slice.iter().map(|r| r.model_name.as_str()).collect();
        let algos: Vec<&str> = slice.iter().map(|r| r.algorithm.as_str()).collect();
        let r2s: Vec<Option<f64>> = slice.iter().map(|r| r.r_squared).collect();
        let mses: Vec<Option<f64>> = slice.iter().map(|r| r.mse).collect();
        let ns: Vec<i32> = slice.iter().map(|r| r.n_samples).collect();
        let nfs: Vec<i32> = slice.iter().map(|r| r.n_features).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(names)),
                Arc::new(StringArray::from(algos)),
                Arc::new(Float64Array::from(r2s)),
                Arc::new(Float64Array::from(mses)),
                Arc::new(Int32Array::from(ns)),
                Arc::new(Int32Array::from(nfs)),
            ],
        )?;

        record_batch_to_duckdb_data_chunk(&batch, output)?;
        *cursor += chunk_size;

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        None
    }
}
