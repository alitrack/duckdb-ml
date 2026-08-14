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

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};
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
                    labels.get(idx).cloned().unwrap_or_else(|| p.to_string())
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

/// ml_ols(y_json, x1_json, x2_json, ...) → VARCHAR
/// Multivariate OLS fit over JSON column arrays — the vscalar equivalent of a
/// regression aggregate. Callers collect columns with the built-in `list()`
/// aggregate and pass them as JSON (subqueries are allowed in scalar-function
/// arguments, so this works over any table):
///
///   SELECT ml_ols(
///       (SELECT to_json(list("y"))  FROM t),
///       (SELECT to_json(list("x1")) FROM t),
///       (SELECT to_json(list("x2")) FROM t));
///
/// Returns a JSON object:
///   {"coefficients":[b1,b2,...], "intercept":b0, "r_squared":.., "mse":..,
///    "n_samples":N, "n_features":K}
pub struct OlsFn;

/// ml_smote(x_json, y_json, k, dup_ratio) → VARCHAR
/// SMOTE oversampling for the minority class. Returns JSON
/// {"x": [[...], ...], "y": [labels], "minority": label, "before": n,
///  "after": m} of synthetic samples. Deterministic (local fixed-seed PRNG).
pub struct SmoteFn;

impl VArrowScalar for SmoteFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let x_json = col_str(&input, 0)?;
        let y_json = col_str(&input, 1)?;
        let k_arr = input
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("col 2 is not DOUBLE")?;
        let dr_arr = input
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("col 3 is not DOUBLE")?;
        let k = k_arr.value(0);
        let dup_ratio = dr_arr.value(0);

        let x: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| format!("Invalid features JSON: {e}"))?;
        let y: Vec<f64> = serde_json::from_str(y_json)
            .map_err(|e| format!("Invalid labels JSON: {e}"))?;

        let result = crate::train::smote::smote(&x, &y, k as usize, dup_ratio);
        let out_json = serde_json::json!({
            "x": result.synthetic_x,
            "y": result.synthetic_y,
            "minority": result.minority_label,
            "before": result.total_before,
            "after": result.total_after,
        })
        .to_string();

        let out = StringArray::from(vec![Some(out_json); n]);
        Ok(Arc::new(out))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![DataType::Utf8, DataType::Utf8, DataType::Float64, DataType::Float64],
            DataType::Utf8,
        )]
    }
}



/// ml_voting(model_names_json, features_json, mode) → VARCHAR
/// Ensemble vote over registered models: mode 'hard' = majority (rounded
/// labels), 'mean' = average. Returns JSON {"votes":[...],"result":v}.
pub struct VotingFn;

impl VArrowScalar for VotingFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let names_json = col_str(&input, 0)?;
        let x_json = col_str(&input, 1)?;
        let mode_str = col_str(&input, 2)?;

        let names: Vec<String> = serde_json::from_str(names_json)
            .map_err(|e| format!("Invalid model names JSON: {e}"))?;
        let x: Vec<f64> = serde_json::from_str(x_json)
            .map_err(|e| format!("Invalid features JSON: {e}"))?;
        let mode = crate::train::voting::VotingMode::parse(mode_str)
            .ok_or_else(|| format!("Invalid voting mode: '{mode_str}' (hard|mean)"))?;

        let votes = crate::train::voting::member_predictions(&names, &x)
            .map_err(|e| e)?;
        let result = crate::train::voting::vote(&names, &x, mode).map_err(|e| e)?;
        let out_json = serde_json::json!({
            "votes": votes,
            "result": result,
            "mode": mode_str,
        })
        .to_string();

        let out = StringArray::from(vec![Some(out_json); n]);
        Ok(Arc::new(out))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![DataType::Utf8, DataType::Utf8, DataType::Utf8],
            DataType::Utf8,
        )]
    }
}

/// ml_km_train(model, time_json, event_json) → VARCHAR
/// Kaplan-Meier survival curve training (feature-less). predict() returns the
/// median survival time.
pub struct KmTrainFn;

impl VArrowScalar for KmTrainFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let model = col_str(&input, 0)?;
        let time = col_str(&input, 1)?;
        let event = col_str(&input, 2)?;

        crate::train::table_fn::km_train_and_register(model, time, event)?;

        let out = StringArray::from(vec![Some(model.to_string()); n]);
        Ok(Arc::new(out))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![DataType::Utf8; 3],
            DataType::Utf8,
        )]
    }
}

/// ml_cox_train(model, time_json, event_json, features_json, params_json) → VARCHAR
/// Cox proportional hazards training with separate time/event arrays.
pub struct CoxTrainFn;

impl VArrowScalar for CoxTrainFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let model = col_str(&input, 0)?;
        let time = col_str(&input, 1)?;
        let event = col_str(&input, 2)?;
        let features = col_str(&input, 3)?;
        let params = col_str(&input, 4)?;

        crate::train::table_fn::cox_train_and_register(model, time, event, features, params)?;

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

fn parse_f64_json(s: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    serde_json::from_str(s).map_err(|e| format!("Invalid JSON array '{s}': {e}").into())
}

/// Shared logic — takes the raw JSON string args (col 0 = y, rest = feature
/// columns), transposes to rows, fits OLS, returns the JSON result string.
pub fn ols_from_json(args: &[&str]) -> Result<String, Box<dyn Error>> {
    if args.len() < 2 {
        return Err("ml_ols requires at least 2 arguments: y, x1[, x2, ...]".into());
    }
    let y = parse_f64_json(args[0])?;
    let n = y.len();
    if n == 0 {
        return Err("ml_ols: empty target array".into());
    }
    let n_features = args.len() - 1;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n_features);
    for (i, arg) in args.iter().enumerate().skip(1) {
        let col = parse_f64_json(arg)?;
        if col.len() != n {
            return Err(format!(
                "ml_ols length mismatch: y has {n} samples, x{} has {}",
                i,
                col.len()
            )
            .into());
        }
        cols.push(col);
    }
    // Column-major → row-major for the trainer
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|r| cols.iter().map(|c| c[r]).collect())
        .collect();

    let result = crate::train::linear::train(&rows, &y, 0.0)?;
    let n_coeffs = result.coefficients.len();
    let json = serde_json::json!({
        "coefficients": &result.coefficients[..n_coeffs - 1], // exclude intercept
        "intercept": result.intercept,
        "r_squared": result.r_squared,
        "mse": result.mse,
        "n_samples": result.num_samples,
        "n_features": n_features,
    });
    Ok(json.to_string())
}

impl VArrowScalar for OlsFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let mut args: Vec<String> = Vec::with_capacity(input.num_columns());
        for idx in 0..input.num_columns() {
            args.push(col_str(&input, idx)?.to_string());
        }
        let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let json = ols_from_json(&arg_refs)?;
        let out = StringArray::from(vec![Some(json); n]);
        Ok(Arc::new(out))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::variadic(
            DataType::Utf8,
            DataType::Utf8,
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_perfect_fit() {
        // y = 2 + 3x on x = [1,2,3,4]
        let json = ols_from_json(&["[5.0,8.0,11.0,14.0]", "[1.0,2.0,3.0,4.0]"]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!((v["coefficients"][0].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((v["intercept"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!((v["r_squared"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(v["n_features"], 1);
        assert_eq!(v["n_samples"], 4);
    }

    #[test]
    fn test_ols_multivariate() {
        // y = 1 + 2*x1 + 3*x2 (checked: row1 1+2+3=6, row2 1+4+12=17, row3 1+2+9=12,
        // row4 1+6+18=25, row5 1+4+24=29)
        let y = "[6.0, 17.0, 12.0, 25.0, 29.0]";
        let x1 = "[1.0, 2.0, 1.0, 3.0, 2.0]";
        let x2 = "[1.0, 4.0, 3.0, 6.0, 8.0]";
        let json = ols_from_json(&[y, x1, x2]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let c: Vec<f64> = v["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert!((c[0] - 2.0).abs() < 1e-6, "b1={}", c[0]);
        assert!((c[1] - 3.0).abs() < 1e-6, "b2={}", c[1]);
        assert!((v["intercept"].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((v["r_squared"].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(v["n_features"], 2);
    }

    #[test]
    fn test_ols_noisy_r2_between_0_and_1() {
        let y = "[2.0, 3.5, 3.0, 5.0, 4.5]";
        let x = "[1.0, 2.0, 2.5, 4.0, 3.5]";
        let json = ols_from_json(&[y, x]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let r2 = v["r_squared"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&r2), "r2={r2}");
    }

    #[test]
    fn test_ols_length_mismatch_errors() {
        let err = ols_from_json(&["[1.0,2.0]", "[1.0,2.0,3.0]"]).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn test_ols_requires_two_args() {
        let err = ols_from_json(&["[1.0,2.0]"]).unwrap_err();
        assert!(err.to_string().contains("at least 2"));
    }
}
