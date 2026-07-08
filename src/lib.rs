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
