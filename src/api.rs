use crate::model::global_registry;
use duckdb::{
    Result,
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
};
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};

#[repr(C)]
pub struct LInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct LBindData {
    names: Vec<String>,
    algorithms: Vec<String>,
    r_squared: Vec<Option<f64>>,
    mse: Vec<Option<f64>>,
}

pub struct ListModelsFn;

impl VTab for ListModelsFn {
    type BindData = LBindData;
    type InitData = LInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let registry = global_registry();
        let model_names = registry.list();

        let mut names = Vec::new();
        let mut algorithms = Vec::new();
        let mut r_squared = Vec::new();
        let mut mse = Vec::new();

        for name in &model_names {
            if let Some(model) = registry.get(name) {
                names.push(name.clone());
                algorithms.push(model.algorithm().to_string());
                r_squared.push(model.metadata().r_squared);
                mse.push(model.metadata().mse);
            }
        }

        bind.add_result_column("name", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("algorithm", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column(
            "created_at",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column("status", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("r_squared", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("mse", LogicalTypeHandle::from(LogicalTypeId::Double));

        Ok(LBindData {
            names,
            algorithms,
            r_squared,
            mse,
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(LInitData {
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
        let n = bind.names.len().min(2048);

        use arrow::array::{Float64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let names: Vec<&str> = bind.names[..n].iter().map(|s| s.as_str()).collect();
        let algs: Vec<&str> = bind.algorithms[..n].iter().map(|s| s.as_str()).collect();
        let empty_str = "".to_string();
        let empty = vec![&empty_str; n];

        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("algorithm", DataType::Utf8, false),
            Field::new("created_at", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("r_squared", DataType::Float64, true),
            Field::new("mse", DataType::Float64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(names)),
                Arc::new(StringArray::from(algs)),
                Arc::new(StringArray::from(
                    empty.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    empty.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(bind.r_squared[..n].to_vec())),
                Arc::new(Float64Array::from(bind.mse[..n].to_vec())),
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
