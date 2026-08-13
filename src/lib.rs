pub mod assoc_rules;
pub mod automl;
pub mod cross_validate;
pub mod deploy;
pub mod embed;
pub mod load;
pub mod metrics;
pub mod model;
pub mod predict;
pub mod scalar;
pub mod snapshot;
pub mod storage;
pub mod train;

mod api;

use duckdb::{duckdb_entrypoint_c_api, Connection, Result};
use std::error::Error;

#[duckdb_entrypoint_c_api(ext_name = "ml")]
pub unsafe fn ml_init(con: Connection) -> Result<(), Box<dyn Error>> {
    log::info!("duckdb_ml v{} initializing", env!("CARGO_PKG_VERSION"));

    // Ensure storage tables exist (v0.9 + v0.10 schema)
    storage::ensure_tables(&con)?;

    // v0.9 functions
    con.register_table_function::<predict::PredictFn>("ml_predict")?;
    con.register_table_function::<predict::batch::PredictBatchFn>("ml_predict_batch")?;
    con.register_table_function::<train::table_fn::TrainFn>("ml_train")?;
    con.register_table_function::<load::LoadXgbFn>("ml_load_xgboost")?;
    #[cfg(feature = "onnx")]
    con.register_table_function::<load::LoadOnnxFn>("ml_load_onnx")?;
    con.register_table_function::<api::ListModelsFn>("ml_list_models")?;

    // v0.10: version management + AutoML
    con.register_table_function::<deploy::DeployFn>("ml_deploy")?;
    con.register_table_function::<automl::CompareFn>("ml_compare")?;

    // v0.13: data version tracking
    con.register_table_function::<snapshot::SnapshotFn>("ml_snapshot")?;
    con.register_table_function::<snapshot::ListSnapshotsFn>("ml_list_snapshots")?;

    // Scalar variants — subquery-friendly params for pipeline SQL (DuckFlow).
    // Table-function params can't contain subqueries; scalar params can.
    con.register_scalar_function::<scalar::TrainModelFn>("ml_train_model")?;
    con.register_scalar_function::<scalar::PredictBatchValueFn>("ml_predict_batch_value")?;
    con.register_scalar_function::<scalar::OlsFn>("ml_ols")?;

    // v0.14: embedding capability (AD-001)
    #[cfg(feature = "onnx")]
    con.register_scalar_function::<embed::EmbedFn>("ml_embed")?;
    con.register_scalar_function::<embed::SimilarityFn>("ml_similarity_value")?;
    con.register_scalar_function::<assoc_rules::AssocRulesFn>("ml_assoc_rules")?;
    con.register_scalar_function::<metrics::MetricsFn>("ml_metrics")?;
    con.register_scalar_function::<cross_validate::CrossValidateFn>("ml_cross_validate")?;

    log::info!("duckdb_ml initialized successfully");
    Ok(())
}

#[cfg(test)]
mod e2e_tests {
    use crate::model::{global_registry, Algorithm, MlModel};
    use crate::train;
    use std::sync::Arc;

    #[test]
    fn e2e_train_predict() {
        let registry = global_registry();
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
            vec![6.0, 5.0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 2.0 * xi[1] + 1.0).collect();
        let test = vec![3.0, 3.0];

        let models = [
            (
                "e2e_lin",
                Algorithm::LinearRegression,
                std::collections::HashMap::new(),
            ),
            (
                "e2e_rf",
                Algorithm::RandomForestRegressor,
                std::collections::HashMap::from([
                    ("n_estimators".into(), 5.0),
                    ("max_depth".into(), 2.0),
                ]),
            ),
            (
                "e2e_xgb",
                Algorithm::XGBoostRegression,
                std::collections::HashMap::from([
                    ("n_estimators".into(), 5.0),
                    ("learning_rate".into(), 0.3),
                    ("max_depth".into(), 2.0),
                ]),
            ),
        ];

        eprintln!("starting loop");
        for (name, algo, params) in &models {
            let result = train::train(*algo, &x, &y, params).unwrap();
            let model: Arc<dyn MlModel> = match *algo {
                Algorithm::LinearRegression => Arc::new(crate::model::linear::LinearModel::new(
                    result.coefficients,
                    result.num_samples,
                    result.r_squared,
                    result.mse,
                    0.0,
                )),
                Algorithm::RandomForestRegressor => Arc::new(
                    crate::model::tree::ForestModel::deserialize(&result.model_blob.unwrap())
                        .unwrap(),
                ),
                Algorithm::XGBoostRegression => Arc::new(
                    crate::model::xgboost::XgbModelWrapper::new(result.model_blob.unwrap())
                        .unwrap(),
                ),
                _ => unreachable!(),
            };
            eprintln!("inserted {name}");
            registry.insert(name.to_string(), model);
        }

        for (name, _, _) in &models {
            eprintln!("getting {name}");
            let model = registry.get(name).unwrap();
            eprintln!("predicting {name}");
            let pred = model.predict(&test).unwrap();
            assert!(pred.is_finite(), "{name}: {pred}");
        }
    }
}
