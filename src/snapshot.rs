//! ml_snapshot table function — register data version snapshots
//!
//! Tracks training data lineage: what data was used to train each model version.
//! Snapshots are stored in-memory (global registry) since DuckDB vtab has no Connection.
//!
//! SQL: SELECT * FROM ml_snapshot('model_name', 'relation_name',
//!         n_features, n_samples, 'target_col', 'feature_columns_json', 'data_hash')
//!
//! Query snapshots: SELECT * FROM ml_list_snapshots('model_name')

use crate::model::global_registry;
use crate::model::registry::DataSnapshot;
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::sync::{
    Arc as StdArc,
    atomic::{AtomicBool, Ordering},
};

#[repr(C)]
pub struct SInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct SBindData {
    model_name: String,
    relation_name: String,
    n_features: i32,
    n_samples: i32,
    target_column: String,
    feature_columns: String,
    data_hash: String,
    success: bool,
    error_msg: String,
}

pub struct SnapshotFn;

impl VTab for SnapshotFn {
    type BindData = SBindData;
    type InitData = SInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 7 {
            return Err(
                "ml_snapshot requires: model_name, relation_name, n_features, n_samples, \
                 target_column, feature_columns_json, data_hash"
                    .into(),
            );
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let relation_name: String = bind.get_parameter(1).to_string();
        let n_features: i32 = bind.get_parameter(2).to_string().parse()?;
        let n_samples: i32 = bind.get_parameter(3).to_string().parse()?;
        let target_column: String = bind.get_parameter(4).to_string();
        let feature_columns_json: String = bind.get_parameter(5).to_string();
        let data_hash: String = bind.get_parameter(6).to_string();

        let feature_columns: Vec<String> = serde_json::from_str(&feature_columns_json)
            .map_err(|e| format!("Invalid feature_columns JSON: {e}"))?;

        let snap = DataSnapshot {
            model_name: model_name.clone(),
            relation_name: relation_name.clone(),
            n_features: n_features as usize,
            n_samples: n_samples as usize,
            target_column: target_column.clone(),
            feature_columns,
            data_hash: data_hash.clone(),
        };

        global_registry().add_snapshot(snap);

        bind.add_result_column(
            "model_name",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column(
            "relation_name",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column(
            "n_features",
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        );
        bind.add_result_column("n_samples", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column(
            "target_column",
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        );
        bind.add_result_column("data_hash", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("success", LogicalTypeHandle::from(LogicalTypeId::Boolean));

        Ok(SBindData {
            model_name,
            relation_name,
            n_features,
            n_samples,
            target_column,
            feature_columns: feature_columns_json,
            data_hash,
            success: true,
            error_msg: String::new(),
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(SInitData {
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

        use arrow::array::{BooleanArray, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;

        let schema = StdArc::new(Schema::new(vec![
            Field::new("model_name", DataType::Utf8, false),
            Field::new("relation_name", DataType::Utf8, false),
            Field::new("n_features", DataType::Int32, false),
            Field::new("n_samples", DataType::Int32, false),
            Field::new("target_column", DataType::Utf8, false),
            Field::new("data_hash", DataType::Utf8, false),
            Field::new("success", DataType::Boolean, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                StdArc::new(StringArray::from(vec![bind.model_name.as_str()])),
                StdArc::new(StringArray::from(vec![bind.relation_name.as_str()])),
                StdArc::new(Int32Array::from(vec![bind.n_features])),
                StdArc::new(Int32Array::from(vec![bind.n_samples])),
                StdArc::new(StringArray::from(vec![bind.target_column.as_str()])),
                StdArc::new(StringArray::from(vec![bind.data_hash.as_str()])),
                StdArc::new(BooleanArray::from(vec![bind.success])),
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

/// Compute a SHA-256 hash of training data for snapshot identification.
/// Compute a SHA-256 hash of training data for snapshot identification.
pub fn hash_training_data(x: &[Vec<f64>], y: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for row in x {
        for &v in row {
            hasher.update(v.to_le_bytes());
        }
    }
    for &v in y {
        hasher.update(v.to_le_bytes());
    }
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::global_registry;

    #[test]
    fn test_hash_training_data() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![5.0, 6.0];
        let h1 = hash_training_data(&x, &y);
        assert_eq!(h1.len(), 64);
        // Same data → same hash
        let h2 = hash_training_data(&x, &y);
        assert_eq!(h1, h2);
        // Different data → different hash
        let y2 = vec![7.0, 6.0];
        let h3 = hash_training_data(&x, &y2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_snapshot_registry() {
        let snap = DataSnapshot {
            model_name: "my_model".into(),
            relation_name: "my_table".into(),
            n_features: 4,
            n_samples: 250,
            target_column: "target".into(),
            feature_columns: vec!["x1".into(), "x2".into(), "x3".into(), "x4".into()],
            data_hash: "abc123def".into(),
        };
        global_registry().add_snapshot(snap);

        let snaps = global_registry().list_snapshots("my_model");
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].relation_name, "my_table");
        assert_eq!(snaps[0].n_features, 4);
        assert_eq!(snaps[0].n_samples, 250);

        let empty = global_registry().list_snapshots("nonexistent");
        assert!(empty.is_empty());
    }
}
