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

    // Ensure storage tables exist
    storage::ensure_tables(&con)?;

    // Register table functions
    con.register_table_function::<predict::PredictFn>("ml_predict")?;
    con.register_table_function::<train::table_fn::TrainFn>("ml_train")?;
    con.register_table_function::<api::ListModelsFn>("ml_list_models")?;

    log::info!("duckdb_ml initialized successfully");
    Ok(())
}
