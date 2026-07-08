//! ml_deploy table function — deploy model versions with strategies
//!
//! SQL: SELECT * FROM ml_deploy('model_name', 'strategy', 'model_key')
//!
//! Parameters:
//!   0: model_name (VARCHAR) — the project/model family name
//!   1: strategy (VARCHAR) — 'best_score' | 'most_recent' | 'rollback'
//!   2: model_key (VARCHAR, optional) — exact registry key for 'most_recent'
//!
//! Returns: model_name, model_key, algorithm, strategy, deployed

use crate::model::global_registry;
use duckdb::{
    Result,
    core::{DataChunkHandle, LogicalTypeHandle},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
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
    model_key: String,
    algorithm: String,
    strategy: String,
    success: bool,
    error_msg: String,
}

pub struct DeployFn;

impl VTab for DeployFn {
    type BindData = DBindData;
    type InitData = DInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 2 {
            return Err("ml_deploy requires: model_name, strategy [, model_key]".into());
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let strategy: String = bind.get_parameter(1).to_string();
        let model_key: String = if n_params >= 3 {
            bind.get_parameter(2).to_string()
        } else {
            model_name.clone()
        };

        let registry = global_registry();

        match registry.deploy(&model_name, &strategy, &model_key) {
            Ok(deployment) => {
                let algo = registry
                    .get(&deployment.model_key)
                    .map(|m| m.algorithm().to_string())
                    .unwrap_or_default();

                Ok(DBindData {
                    model_name,
                    model_key: deployment.model_key,
                    algorithm: algo,
                    strategy,
                    success: true,
                    error_msg: String::new(),
                })
            }
            Err(e) => Ok(DBindData {
                model_name,
                model_key: String::new(),
                algorithm: String::new(),
                strategy,
                success: false,
                error_msg: e,
            }),
        }
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

        use arrow::array::{BooleanArray, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("model_name", DataType::Utf8, false),
            Field::new("model_key", DataType::Utf8, false),
            Field::new("algorithm", DataType::Utf8, false),
            Field::new("strategy", DataType::Utf8, false),
            Field::new("success", DataType::Boolean, false),
            Field::new("error", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![bind.model_name.as_str()])),
                Arc::new(StringArray::from(vec![bind.model_key.as_str()])),
                Arc::new(StringArray::from(vec![bind.algorithm.as_str()])),
                Arc::new(StringArray::from(vec![bind.strategy.as_str()])),
                Arc::new(BooleanArray::from(vec![bind.success])),
                Arc::new(StringArray::from(vec![if bind.error_msg.is_empty() {
                    None
                } else {
                    Some(bind.error_msg.as_str())
                }])),
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
