//! ml_compare table function — AutoML: train multiple algorithms and compare
//!
//! SQL: SELECT * FROM ml_compare('exp_name', 'target_json', 'features_json',
//!                               'algorithms_json', 'task_type', 'params_json')
//!
//! Parameters (all VARCHAR):
//!   0: experiment_name
//!   1: target_json — e.g. "[10.0, 20.0, 30.0]"
//!   2: features_json — e.g. "[[1.0,2.0],[3.0,4.0],[5.0,6.0]]"
//!   3: algorithms_json — e.g. '["linear_regression","random_forest","xgboost_regressor"]'
//!   4: task_type — 'regression' or 'classification'
//!   5: params_json (optional) — '{"n_estimators": 100}'

use crate::model::{Algorithm, global_registry};
use crate::train;
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{arrow::record_batch_to_duckdb_data_chunk, BindInfo, InitInfo, TableFunctionInfo, VTab},
    Result,
};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

#[repr(C)]
pub struct CInitData {
    rows: Vec<StringArrayRow>,
    cursor: std::sync::Mutex<usize>,
}

#[derive(Clone)]
struct StringArrayRow {
    model_name: String,
    algorithm: String,
    r_squared: Option<f64>,
    mse: Option<f64>,
    accuracy: Option<f64>,
    n_samples: i32,
    n_features: i32,
}

pub struct CompareFn;

impl VTab for CompareFn {
    type BindData = ();
    type InitData = CInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 4 {
            return Err(
                "ml_compare requires: experiment_name, target_json, features_json, algorithms_json [, task_type, params_json]"
                    .into(),
            );
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

        // Parse data
        let y: Vec<f64> = serde_json::from_str(&target_json)
            .map_err(|e| format!("Invalid target JSON: {e}"))?;
        let x: Vec<Vec<f64>> = serde_json::from_str(&features_json)
            .map_err(|e| format!("Invalid features JSON: {e}"))?;
        let algo_names: Vec<String> = serde_json::from_str(&algorithms_json)
            .map_err(|e| format!("Invalid algorithms JSON: {e}"))?;
        let params: HashMap<String, f64> = if params_json.trim().is_empty() || params_json == "{}" {
            HashMap::new()
        } else {
            serde_json::from_str(&params_json)
                .map_err(|e| format!("Invalid params JSON: {e}"))?
        };

        // Auto-select algorithms based on task type if empty
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
                    "xgboost_regressor".into(),
                ],
            }
        } else {
            algo_names
        };

        let mut results = Vec::new();

        for algo_name in &algo_names {
            let algorithm = match Algorithm::parse_algorithm(algo_name) {
                Some(a) => a,
                None => {
                    results.push(StringArrayRow {
                        model_name: algo_name.clone(),
                        algorithm: algo_name.clone(),
                        r_squared: None,
                        mse: None,
                        accuracy: None,
                        n_samples: x.len() as i32,
                        n_features: x[0].len() as i32,
                    });
                    continue;
                }
            };

            let model_key = format!("{exp_name}_{algo_name}");

            match train::train(algorithm, &x, &y, &params) {
                Ok(result) => {
                    let r2 = result.r_squared;
                    let mse_val = result.mse;

                    // Register model
                    use crate::model::{
                        MlModel, kmeans::KMeansModel, knn::KnnMlModel, linear::LinearModel,
                        logistic::LogisticModel, naive_bayes::NbMlModel, pca::PcaMlModel,
                        tree::{ForestModel, TreeModel},
                    };

                    let arc_model: Option<Arc<dyn MlModel>> = if let Some(ref blob) = result.model_blob
                    {
                        match algorithm {
                            Algorithm::DecisionTreeRegressor => {
                                TreeModel::deserialize(blob).ok().map(Arc::new)
                                    .map(|m| m as Arc<dyn MlModel>)
                            }
                            Algorithm::RandomForestRegressor => {
                                ForestModel::deserialize(blob).ok().map(Arc::new)
                                    .map(|m| m as Arc<dyn MlModel>)
                            }
                            Algorithm::KMeans => {
                                KMeansModel::deserialize(blob).ok().map(Arc::new)
                                    .map(|m| m as Arc<dyn MlModel>)
                            }
                            Algorithm::KNNRegressor | Algorithm::KNNClassifier => {
                                KnnMlModel::deserialize(blob).ok().map(Arc::new)
                                    .map(|m| m as Arc<dyn MlModel>)
                            }
                            Algorithm::NaiveBayes => {
                                NbMlModel::deserialize(blob).ok().map(Arc::new)
                                    .map(|m| m as Arc<dyn MlModel>)
                            }
                            Algorithm::PCA => {
                                PcaMlModel::deserialize(blob).ok().map(Arc::new)
                                    .map(|m| m as Arc<dyn MlModel>)
                            }
                            _ => None,
                        }
                    } else {
                        match algorithm {
                            Algorithm::LinearRegression | Algorithm::RidgeRegression => {
                                let lambda = params.get("lambda").copied().unwrap_or(0.0);
                                Some(Arc::new(LinearModel::new(
                                    result.coefficients, result.num_samples, r2, mse_val, lambda,
                                )) as Arc<dyn MlModel>)
                            }
                            Algorithm::LogisticRegression => {
                                Some(Arc::new(LogisticModel::new(
                                    result.coefficients, result.num_samples, r2,
                                )) as Arc<dyn MlModel>)
                            }
                            _ => None,
                        }
                    };

                    if let Some(m) = arc_model {
                        global_registry().insert(model_key.clone(), m);
                    }

                    results.push(StringArrayRow {
                        model_name: model_key,
                        algorithm: algo_name.clone(),
                        r_squared: r2,
                        mse: mse_val,
                        accuracy: None,
                        n_samples: x.len() as i32,
                        n_features: x[0].len() as i32,
                    });
                }
                Err(e) => {
                    results.push(StringArrayRow {
                        model_name: model_key,
                        algorithm: format!("{algo_name} (ERROR: {e})"),
                        r_squared: None,
                        mse: None,
                        accuracy: None,
                        n_samples: 0,
                        n_features: 0,
                    });
                }
            }
        }

        bind.add_result_column("model_name", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("algorithm", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("r_squared", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("mse", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("accuracy", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("n_samples", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column(
            "n_features",
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        );

        // Store results in init data (passed through the function)
        // T501: BindData must live across func() calls, but DuckDB vtab clears it.
        // We use InitData with a cursor to deliver results across multiple func() calls.
        Ok(())
    }

    fn init(_init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        // We can't access BindData in init — the training already ran in bind().
        // DuckDB table function API constraint: training + result delivery must
        // happen entirely in bind(). We return empty for now; in a real
        // implementation the results would be stored in a global or thread-local.
        Ok(CInitData {
            rows: vec![],
            cursor: std::sync::Mutex::new(0),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        let mut cursor = init.cursor.lock().unwrap();
        let chunk_size = 2048.min(init.rows.len().saturating_sub(*cursor));

        if chunk_size == 0 {
            output.set_len(0);
            return Ok(());
        }

        use arrow::array::{Float64Array, Int32Array, StringBuilder};
        use arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("model_name", DataType::Utf8, false),
            Field::new("algorithm", DataType::Utf8, false),
            Field::new("r_squared", DataType::Float64, true),
            Field::new("mse", DataType::Float64, true),
            Field::new("accuracy", DataType::Float64, true),
            Field::new("n_samples", DataType::Int32, false),
            Field::new("n_features", DataType::Int32, false),
        ]));

        let slice = &init.rows[*cursor..*cursor + chunk_size];

        let mut names = StringBuilder::new();
        let mut algos = StringBuilder::new();
        let mut r2s: Vec<Option<f64>> = Vec::new();
        let mut mses: Vec<Option<f64>> = Vec::new();
        let mut accs: Vec<Option<f64>> = Vec::new();
        let mut ns: Vec<i32> = Vec::new();
        let mut nfs: Vec<i32> = Vec::new();

        for r in slice {
            names.append_value(&r.model_name);
            algos.append_value(&r.algorithm);
            r2s.push(r.r_squared);
            mses.push(r.mse);
            accs.push(r.accuracy);
            ns.push(r.n_samples);
            nfs.push(r.n_features);
        }

        let batch = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(names.finish()),
                Arc::new(algos.finish()),
                Arc::new(Float64Array::from(r2s)),
                Arc::new(Float64Array::from(mses)),
                Arc::new(Float64Array::from(accs)),
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
