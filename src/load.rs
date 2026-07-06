//! External model loaders — ml_load_onnx, ml_load_xgboost
//!
//! Load pre-trained models from files into the global registry for ml_predict.

use crate::model::MlModel;
use crate::model::global_registry;
use duckdb::{
    Result,
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab, arrow::record_batch_to_duckdb_data_chunk},
};
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// ── Shared infrastructure ──

#[repr(C)]
pub struct LInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct LBindData {
    model_name: String,
    algorithm: String,
    status: String,
    num_features: i32,
}

fn make_output(
    func: &TableFunctionInfo<impl VTab<InitData = LInitData, BindData = LBindData>>,
    output: &mut DataChunkHandle,
) -> Result<(), Box<dyn Error>> {
    let init = func.get_init_data();
    if init.done.load(Ordering::Relaxed) {
        output.set_len(0);
        return Ok(());
    }
    let bind = func.get_bind_data();

    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(Schema::new(vec![
        Field::new("model_name", DataType::Utf8, false),
        Field::new("algorithm", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("num_features", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![bind.model_name.as_str()])),
            Arc::new(StringArray::from(vec![bind.algorithm.as_str()])),
            Arc::new(StringArray::from(vec![bind.status.as_str()])),
            Arc::new(Int32Array::from(vec![Some(bind.num_features)])),
        ],
    )?;
    record_batch_to_duckdb_data_chunk(&batch, output)?;
    init.done.store(true, Ordering::Relaxed);
    Ok(())
}

fn add_result_columns(bind: &BindInfo) {
    bind.add_result_column(
        "model_name",
        LogicalTypeHandle::from(LogicalTypeId::Varchar),
    );
    bind.add_result_column("algorithm", LogicalTypeHandle::from(LogicalTypeId::Varchar));
    bind.add_result_column("status", LogicalTypeHandle::from(LogicalTypeId::Varchar));
    bind.add_result_column(
        "num_features",
        LogicalTypeHandle::from(LogicalTypeId::Integer),
    );
}

// ── ml_load_onnx ──

pub struct LoadOnnxFn;

impl VTab for LoadOnnxFn {
    type BindData = LBindData;
    type InitData = LInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 2 {
            return Err("ml_load_onnx requires: model_name, file_path [, num_features]".into());
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let file_path: String = bind.get_parameter(1).to_string();

        let n_features: usize = if n_params >= 3 {
            bind.get_parameter(2).to_string().parse::<usize>()?
        } else {
            0 // will be inferred from first input-shape lookup at predict time
        };

        #[cfg(feature = "onnx")]
        {
            let model = crate::model::onnx::OnnxModel::new(&file_path, n_features)?;
            let algo = model.algorithm().to_string();
            let nf = model.metadata().num_features as i32;
            global_registry().insert(model_name.clone(), Arc::new(model));

            add_result_columns(bind);
            Ok(LBindData {
                model_name,
                algorithm: algo,
                status: "loaded".into(),
                num_features: nf,
            })
        }

        #[cfg(not(feature = "onnx"))]
        {
            let _ = (model_name, file_path, n_features);
            Err("ONNX support not compiled (enable 'onnx' feature)".into())
        }
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
        make_output(func, output)
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        None
    }
}

// ── ml_load_xgboost ──

pub struct LoadXgbFn;

impl VTab for LoadXgbFn {
    type BindData = LBindData;
    type InitData = LInitData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let n_params = bind.get_parameter_count();
        if n_params < 2 {
            return Err("ml_load_xgboost requires: model_name, file_path".into());
        }

        let model_name: String = bind.get_parameter(0).to_string();
        let file_path: String = bind.get_parameter(1).to_string();

        let json_bytes = std::fs::read(&file_path)
            .map_err(|e| format!("Cannot read XGBoost model file '{file_path}': {e}"))?;

        let model = crate::model::xgboost::XgbModelWrapper::new(json_bytes)?;
        let algo = model.algorithm().to_string();
        let nf = model.metadata().num_features as i32;
        global_registry().insert(model_name.clone(), Arc::new(model));

        add_result_columns(bind);
        Ok(LBindData {
            model_name,
            algorithm: algo,
            status: "loaded".into(),
            num_features: nf,
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
        make_output(func, output)
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::xgboost::XgbModelWrapper;

    /// Minimal valid XGBoost JSON for testing (752 bytes, 1 tree, 2 features)
    fn make_xgb_json() -> String {
        r#"{"version":[2,0,0],"learner":{"gradient_booster":{"name":"gbtree","model":{"gbtree_model_param":{"num_trees":"1","num_features":"2"},"trees":[{"base_weights":[-0.5,0.5,-1.0],"categories":[],"categories_nodes":[],"categories_segments":[],"categories_sizes":[],"default_left":[false,false,false],"id":0,"left_children":[-1,-1,-1],"loss_changes":[0.0,0.0,0.0],"parents":[2147483647,0,0],"right_children":[-1,-1,-1],"split_conditions":[-0.5,0.5,0.0],"split_indices":[0,1,0],"split_type":[0,0,0],"sum_hessian":[0.0,0.0,0.0],"tree_param":{"num_deleted":"0","num_feature":"2","num_nodes":"3","size_leaf_vector":"0"}}],"tree_info":[0,0,0]}},"learner_model_param":{"base_score":"5E-1","num_class":"0","num_feature":"2"},"objective":{"name":"reg:squarederror"}}}"#.to_string()
    }

    #[test]
    fn test_load_xgboost_from_bytes() {
        let json = make_xgb_json();
        let model = XgbModelWrapper::new(json.into_bytes()).unwrap();
        assert_eq!(model.algorithm(), crate::model::Algorithm::XGBoostRegressor);

        global_registry().insert("test_xgb_load".into(), Arc::new(model));
        let m = global_registry().get("test_xgb_load").unwrap();
        let pred = m.predict(&[1.0, 2.0]).unwrap();
        assert!(pred.is_finite(), "pred={pred}");
    }
}
