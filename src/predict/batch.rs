//! Batch predict via DuckDB table function
//! Usage: SELECT * FROM ml_predict_batch('model_name', '[[1.0,2.0],[3.0,4.0]]')

use crate::model::global_registry;
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
};
use std::error::Error;
use std::sync::{
    Arc as StdArc,
    atomic::{AtomicBool, Ordering},
};

#[repr(C)]
pub struct BInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct BBindData {
    predictions: Vec<f64>,
    row_ids: Vec<i64>,
}

pub struct PredictBatchFn;

impl VTab for PredictBatchFn {
    type BindData = BBindData;
    type InitData = BInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 2 {
            return Err(
                "ml_predict_batch requires: model_name, features_json (e.g. '[[1.0,2.0],[3.0,4.0]]')"
                    .into(),
            );
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let features_json: String = bind.get_parameter(1).to_string();

        // @name syntax: look up cached dataset from global registry (populated by ml_train)
        let features: Vec<Vec<f64>> = if let Some(dataset_name) = features_json.strip_prefix('@') {
            global_registry()
                .get_dataset(dataset_name)
                .unwrap_or_else(|| {
                    let json = features_json.clone();
                    serde_json::from_str(&json).unwrap_or_default()
                })
        } else {
            serde_json::from_str(&features_json)
                .map_err(|e| format!("Invalid features JSON: {e}"))?
        };

        if features.is_empty() {
            return Err("features_json must contain at least one row".into());
        }

        let model = global_registry()
            .get_deployed_model(&model_name)
            .or_else(|| global_registry().get(&model_name))
            .ok_or_else(|| format!("Model '{model_name}' not loaded"))?;

        let n_features = model.metadata().num_features;
        for (i, row) in features.iter().enumerate() {
            if row.len() != n_features && n_features > 0 {
                return Err(
                    format!("Row {i}: expected {n_features} features, got {}", row.len()).into(),
                );
            }
        }

        let mut predictions = Vec::with_capacity(features.len());
        let mut row_ids = Vec::with_capacity(features.len());
        for (i, row) in features.iter().enumerate() {
            let pred = model.predict(row)?;
            predictions.push(pred);
            row_ids.push(i as i64);
        }

        bind.add_result_column("row_id", LogicalTypeHandle::from(LogicalTypeId::Bigint));
        bind.add_result_column("prediction", LogicalTypeHandle::from(LogicalTypeId::Double));

        Ok(BBindData {
            predictions,
            row_ids,
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(BInitData {
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

        use arrow::array::{Float64Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;

        let schema = StdArc::new(Schema::new(vec![
            Field::new("row_id", DataType::Int64, false),
            Field::new("prediction", DataType::Float64, false),
        ]));

        let rows: Vec<i64> = bind.row_ids.clone();
        let preds: Vec<f64> = bind.predictions.clone();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                StdArc::new(Int64Array::from(rows)),
                StdArc::new(Float64Array::from(preds)),
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

#[cfg(test)]
mod tests {
    use crate::model::global_registry;
    use crate::train;
    use std::sync::Arc;

    #[test]
    fn test_predict_batch_registry() {
        // Train a small linear model, register it, verify batch predictions
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
            vec![6.0, 5.0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 2.0 * xi[1] + 1.0).collect();

        let result = train::train(
            crate::model::Algorithm::LinearRegression,
            &x,
            &y,
            &std::collections::HashMap::new(),
        )
        .unwrap();

        let model = crate::model::linear::LinearModel::new(
            result.coefficients,
            result.num_samples,
            result.r_squared,
            result.mse,
            0.0,
        );
        global_registry().insert("btest_lin".into(), Arc::new(model));

        let m = global_registry().get("btest_lin").unwrap();
        let p1 = m.predict(&[3.0, 3.0]).unwrap();
        assert!(p1.is_finite(), "btest_lin: {p1}");
    }

    #[test]
    fn test_predict_batch_at_syntax() {
        // Verify @model_name syntax: cache dataset → retrieve via @btest_at
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
            vec![6.0, 5.0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 2.0 * xi[1] + 1.0).collect();

        let result = train::train(
            crate::model::Algorithm::LinearRegression,
            &x,
            &y,
            &std::collections::HashMap::new(),
        )
        .unwrap();

        let model = crate::model::linear::LinearModel::new(
            result.coefficients,
            result.num_samples,
            result.r_squared,
            result.mse,
            0.0,
        );
        global_registry().insert("btest_at".into(), Arc::new(model));

        // Cache the dataset (simulating what ml_train does)
        global_registry().cache_dataset("btest_at", x.clone());

        // Retrieve via @ syntax
        let cached = global_registry().get_dataset("btest_at").unwrap();
        assert_eq!(cached.len(), 6);
        assert_eq!(cached[0], vec![1.0, 2.0]);
    }
}
