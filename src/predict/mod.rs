//! Predict via DuckDB table function
//! Usage: SELECT * FROM ml_predict('model_name', val1, val2, ...)

pub mod batch;

use crate::model::global_registry;
use duckdb::{
    Result,
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
};
use std::error::Error;
use std::sync::{
    Arc as StdArc,
    atomic::{AtomicBool, Ordering},
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
            return Err("ml_predict requires model_name and feature values".into());
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let mut features = Vec::with_capacity((n_params - 1) as usize);
        for i in 1..n_params {
            features.push(bind.get_parameter(i).to_string().parse::<f64>()?);
        }

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
        None
    }
}
