//! ml_deploy table function — deploy model versions with strategies
//!
//! SQL: SELECT * FROM ml_deploy('model_name', 'strategy')
//!
//! Parameters:
//!   0: model_name (VARCHAR)
//!   1: strategy (VARCHAR) — 'best_score' | 'most_recent' | 'rollback'
//!   2: algorithm (VARCHAR, optional) — filter by algorithm for best_score
//!
//! Returns: deployment_id, model_name, model_id, strategy, deployed_at

use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{arrow::record_batch_to_duckdb_data_chunk, BindInfo, InitInfo, TableFunctionInfo, VTab},
    Result,
};
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};

#[repr(C)]
pub struct DInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct DBindData {
    model_name: String,
    model_id: i64,
    strategy: String,
    deployed_at: String,
}

pub struct DeployFn;

impl VTab for DeployFn {
    type BindData = DBindData;
    type InitData = DInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 2 {
            return Err("ml_deploy requires: model_name, strategy [algorithm]".into());
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let strategy: String = bind.get_parameter(1).to_string();
        let algorithm: Option<String> = if n_params >= 3 {
            let a: String = bind.get_parameter(2).to_string();
            if a.is_empty() {
                None
            } else {
                Some(a)
            }
        } else {
            None
        };

        let valid_strategies = ["best_score", "most_recent", "rollback"];
        if !valid_strategies.contains(&strategy.as_str()) {
            return Err(format!(
                "Unknown strategy '{strategy}'. Available: {}",
                valid_strategies.join(", ")
            )
            .into());
        }

        // Use a transient Connection to run the deployment logic
        let mut con = duckdb::Connection::open_in_memory()?;

        // DuckDB vtab doesn't give us a direct Connection, so we use
        // the storage module's connection-agnostic SQL builder.
        // For simplicity, we compute the result inline.
        let (model_id, deployed_at) =
            execute_deploy(&mut con, &model_name, &strategy, algorithm.as_deref())?;

        bind.add_result_column(
            "deployment_id",
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        );
        bind.add_result_column(
            "model_name",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column("model_id", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("strategy", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column(
            "deployed_at",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );

        Ok(DBindData {
            model_name,
            model_id,
            strategy,
            deployed_at,
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(DInitData {
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

        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("deployment_id", DataType::Int64, false),
            Field::new("model_name", DataType::Utf8, false),
            Field::new("model_id", DataType::Int64, false),
            Field::new("strategy", DataType::Utf8, false),
            Field::new("deployed_at", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![bind.model_id])),
                Arc::new(StringArray::from(vec![bind.model_name.as_str()])),
                Arc::new(Int64Array::from(vec![bind.model_id])),
                Arc::new(StringArray::from(vec![bind.strategy.as_str()])),
                Arc::new(StringArray::from(vec![bind.deployed_at.as_str()])),
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

/// Execute the deploy logic against the in-memory DB.
/// In a real setup this would use the main DuckDB connection.
fn execute_deploy(
    con: &mut duckdb::Connection,
    model_name: &str,
    strategy: &str,
    algorithm: Option<&str>,
) -> Result<(i64, String), Box<dyn Error>> {
    // Create tables (in production, these already exist from ensure_tables)
    con.execute_batch(
        "CREATE TABLE IF NOT EXISTS models_v2 (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            algorithm TEXT NOT NULL,
            metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, version)
        );
        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );",
    )?;

    // Find the target model
    let find_sql = match strategy {
        "best_score" => {
            if let Some(algo) = algorithm {
                format!(
                    "SELECT id FROM models_v2
                     WHERE name = '{n}' AND algorithm = '{a}'
                     ORDER BY COALESCE((metrics->>'r_squared')::DOUBLE, (metrics->>'accuracy')::DOUBLE, 0) DESC
                     LIMIT 1",
                    n = model_name.replace('\'', "''"),
                    a = algo.replace('\'', "''"),
                )
            } else {
                format!(
                    "SELECT id FROM models_v2
                     WHERE name = '{n}'
                     ORDER BY COALESCE((metrics->>'r_squared')::DOUBLE, (metrics->>'accuracy')::DOUBLE, 0) DESC
                     LIMIT 1",
                    n = model_name.replace('\'', "''"),
                )
            }
        }
        "most_recent" => {
            format!(
                "SELECT id FROM models_v2 WHERE name = '{n}' ORDER BY created_at DESC LIMIT 1",
                n = model_name.replace('\'', "''"),
            )
        }
        "rollback" => {
            format!(
                "SELECT m.id FROM models_v2 m
                 JOIN (
                     SELECT model_id FROM deployments
                     WHERE model_name = '{n}'
                     ORDER BY created_at DESC LIMIT 1 OFFSET 1
                 ) d ON m.id = d.model_id
                 LIMIT 1",
                n = model_name.replace('\'', "''"),
            )
        }
        _ => return Err(format!("Unknown strategy: {strategy}").into()),
    };

    let mut stmt = con.prepare(&find_sql)?;
    let mut rows = stmt.query([])?;
    let model_id: i64 = if let Some(row) = rows.next()? {
        row.get(0)?
    } else {
        return Err(
            format!("No model found for '{model_name}' with strategy '{strategy}'").into(),
        );
    };

    // Insert deployment
    let insert_sql = format!(
        "INSERT INTO deployments (model_name, model_id, strategy) VALUES ('{n}', {id}, '{s}')",
        n = model_name.replace('\'', "''"),
        id = model_id,
        s = strategy,
    );
    con.execute(&insert_sql, [])?;

    // Get timestamp
    let ts: String = con
        .query_row("SELECT CURRENT_TIMESTAMP::VARCHAR", [], |row| row.get(0))?;

    Ok((model_id, ts))
}
