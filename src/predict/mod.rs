//! Predict via DuckDB table function
//! Usage: SELECT * FROM ml_predict('model_name', '[1.0, 2.0]')

pub mod batch;

use crate::model::global_registry;
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{arrow::record_batch_to_duckdb_data_chunk, BindInfo, InitInfo, TableFunctionInfo, VTab},
    Result,
};
use std::error::Error;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc as StdArc,
};

#[repr(C)]
pub struct PInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct PBindData {
    prediction: f64,
}

pub struct PredictFn;

impl VTab for PredictFn {
    type BindData = PBindData;
    type InitData = PInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 2 {
            return Err(
                "ml_predict requires model_name and features_json (e.g. '[1.0,2.0]')".into(),
            );
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let features_json: String = bind.get_parameter(1).to_string();
        let features: Vec<f64> = serde_json::from_str(&features_json)
            .map_err(|e| format!("Invalid features JSON '{features_json}': {e}"))?;

        // Try global registry first, fall back to storage
        let model = global_registry()
            .get(&model_name)
            .or_else(|| {
                // Try loading from storage (this requires DB access, but registry is primary)
                None // for now: models must be registered via CREATE MODEL
            })
            .ok_or_else(|| format!("Model '{model_name}' not loaded"))?;

        let prediction = model.predict(&features)?;

        bind.add_result_column("prediction", LogicalTypeHandle::from(LogicalTypeId::Double));
        Ok(PBindData { prediction })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(PInitData {
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
        use arrow::array::Float64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        let schema = StdArc::new(Schema::new(vec![Field::new(
            "prediction",
            DataType::Float64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![StdArc::new(Float64Array::from(vec![Some(bind.prediction)]))],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        init.done.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}
