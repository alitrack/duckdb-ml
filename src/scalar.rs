//! Scalar SQL functions — subquery-friendly parameters.
//!
//! DuckDB forbids subqueries inside table-function parameters
//! ("Table function cannot contain subqueries"), but scalar-function parameters
//! may contain them. DuckFlow therefore calls the ML pipeline through scalar
//! functions, computing the JSON payloads from CTEs inline:
//!
//!   SELECT ml_train_model('kf_2', 'decision_tree',
//!       (SELECT to_json(list("income")) FROM _1),
//!       (SELECT to_json(list(["age","score"])) FROM _1), '{}');
//!
//!   SELECT unnest(from_json(
//!       ml_predict_batch_value('kf_2',
//!           (SELECT to_json(list(["age","score"])) FROM _1)),
//!       '["DOUBLE"]')) AS prediction;

use arrow::array::{Array, ArrayRef, StringArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use duckdb::vscalar::arrow::{ArrowFunctionSignature, VArrowScalar};
use std::error::Error;
use std::sync::Arc;

fn col_str(input: &RecordBatch, idx: usize) -> Result<&str, Box<dyn Error>> {
    let arr = input
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("expected VARCHAR column")?;
    if arr.is_null(0) {
        return Err("NULL parameter".into());
    }
    Ok(arr.value(0))
}

/// ml_train_model(model, algo, target_json, features_json, params_json) → VARCHAR
/// Trains + registers a model (side effect), returns the model name.
pub struct TrainModelFn;

impl VArrowScalar for TrainModelFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let model = col_str(&input, 0)?;
        let algo = col_str(&input, 1)?;
        let target = col_str(&input, 2)?;
        let features = col_str(&input, 3)?;
        let params = col_str(&input, 4)?;

        crate::train::table_fn::train_and_register(model, algo, target, features, params)?;

        let out = StringArray::from(vec![Some(model.to_string()); n]);
        Ok(Arc::new(out))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![DataType::Utf8; 5],
            DataType::Utf8,
        )]
    }
}

/// ml_predict_batch_value(model, features_json) → VARCHAR
/// Batch prediction; returns a JSON array of predictions, e.g. '[0.5,0.5]'.
pub struct PredictBatchValueFn;

impl VArrowScalar for PredictBatchValueFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let model = col_str(&input, 0)?;
        let features_json = col_str(&input, 1)?;

        let x: Vec<Vec<f64>> = serde_json::from_str(features_json)
            .map_err(|e| format!("Invalid features JSON '{features_json}': {e}"))?;
        let arc_model = crate::model::global_registry()
            .get(model)
            .ok_or_else(|| format!("Model '{model}' not loaded"))?;

        let mut preds = Vec::with_capacity(x.len());
        for row in &x {
            preds.push(arc_model.predict(row)?);
        }
        // Decode ordinal encodings back to original class labels (string targets);
        // numeric targets pass through as-is.
        let json = if let Some(labels) = crate::model::global_registry().label_map_for(model) {
            let decoded: Vec<String> = preds
                .iter()
                .map(|&p| {
                    let idx = p.round() as usize;
                    labels
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| p.to_string())
                })
                .collect();
            serde_json::to_string(&decoded)?
        } else {
            serde_json::to_string(&preds)?
        };

        let out = StringArray::from(vec![Some(json); n]);
        Ok(Arc::new(out))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![DataType::Utf8, DataType::Utf8],
            DataType::Utf8,
        )]
    }
}
