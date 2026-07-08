pub mod automl;
pub mod deploy;
pub mod load;
pub mod model;
pub mod predict;
pub mod storage;
pub mod train;

mod api;

use duckdb::{Connection, Result, duckdb_entrypoint_c_api};
use std::error::Error;

#[duckdb_entrypoint_c_api(ext_name = "ml")]
pub unsafe fn ml_init(con: Connection) -> Result<(), Box<dyn Error>> {
    log::info!("duckdb_ml v{} initializing", env!("CARGO_PKG_VERSION"));

    // Ensure storage tables exist (v0.9 + v0.10 schema)
    storage::ensure_tables(&con)?;

    // v0.9 functions
    con.register_table_function::<predict::PredictFn>("ml_predict")?;
    con.register_table_function::<train::table_fn::TrainFn>("ml_train")?;
    con.register_table_function::<load::LoadXgbFn>("ml_load_xgboost")?;
    #[cfg(feature = "onnx")]
    con.register_table_function::<load::LoadOnnxFn>("ml_load_onnx")?;
    con.register_table_function::<api::ListModelsFn>("ml_list_models")?;

    // v0.10: version management + AutoML
    con.register_table_function::<deploy::DeployFn>("ml_deploy")?;
    con.register_table_function::<automl::CompareFn>("ml_compare")?;

    log::info!("duckdb_ml initialized successfully");
    Ok(())
}

#[cfg(test)]
mod e2e_tests {
    use crate::model::{Algorithm, MlModel, global_registry};
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
