//! K-fold cross validation over the training pipeline (MADlib `validation`
//! counterpart).
//!
//! SQL surface:
//!
//! ```sql
//! SELECT ml_cross_validate(
//!     'linear_regression',           -- algorithm (see ml_train_model)
//!     '[[1.0,2.0],[2.0,3.0],...]',  -- features (n × d)
//!     '[3.0,5.0,...]',              -- targets
//!     '{"lambda": 0.1}',            -- hyperparameters (or NULL)
//!     5);                           -- folds (optional, default 5)
//! ```
//!
//! Deterministic sequential folds (fold f = indices with i % k == f), so the
//! same input always yields the same result. Returns JSON with per-fold and
//! mean `mse` / `r2` (regression algorithms; classification algorithms whose
//! training result carries no scores report `null`).

use std::error::Error;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use duckdb::vscalar::arrow::{ArrowFunctionSignature, VArrowScalar};
use serde_json::json;

use crate::model::Algorithm;
use crate::train;

/// `ml_cross_validate(algorithm, features_json, targets_json, params_json[, k])`
pub struct CrossValidateFn;

impl VArrowScalar for CrossValidateFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let n_args = input.num_columns();
        if !(4..=5).contains(&n_args) {
            return Err(
                "ml_cross_validate requires 4..5 args: algorithm, features, targets, params[, k]"
                    .into(),
            );
        }

        let algo_col = col_str(&input, 0)?;
        let features_col = col_str(&input, 1)?;
        let targets_col = col_str(&input, 2)?;
        let params_col = col_str(&input, 3)?;

        let k: usize = if n_args >= 5 {
            let c = col_str(&input, 4)?;
            if c.is_null(0) {
                5
            } else {
                c.value(0).parse::<f64>().map(|v| v as usize).unwrap_or(5)
            }
        } else {
            5
        };

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if algo_col.is_null(i) || features_col.is_null(i) || targets_col.is_null(i) {
                out.push(None);
                continue;
            }
            let params = if params_col.is_null(i) {
                "null"
            } else {
                params_col.value(i)
            };
            let json = cross_validate_from_json(
                algo_col.value(i),
                features_col.value(i),
                targets_col.value(i),
                params,
                k,
            )?;
            out.push(Some(json));
        }
        let arr = StringArray::from(out);
        Ok(Arc::new(arr))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![
            ArrowFunctionSignature::exact(
                vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ],
                DataType::Utf8,
            ),
            ArrowFunctionSignature::exact(
                vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ],
                DataType::Utf8,
            ),
        ]
    }
}

/// Run k-fold cross validation; returns the JSON payload.
pub fn cross_validate_from_json(
    algorithm_str: &str,
    features_json: &str,
    targets_json: &str,
    params_json: &str,
    k: usize,
) -> Result<String, String> {
    let algorithm = Algorithm::parse_algorithm(algorithm_str)
        .ok_or_else(|| format!("ml_cross_validate: unknown algorithm '{algorithm_str}'"))?;

    let x: Vec<Vec<f64>> = serde_json::from_str(features_json)
        .map_err(|e| format!("ml_cross_validate: bad features JSON: {e}"))?;
    let y: Vec<f64> = serde_json::from_str(targets_json)
        .map_err(|e| format!("ml_cross_validate: bad targets JSON: {e}"))?;

    if x.is_empty() {
        return Err("ml_cross_validate: empty dataset".into());
    }
    if x.len() != y.len() {
        return Err(format!(
            "ml_cross_validate: length mismatch: {} features vs {} targets",
            x.len(),
            y.len()
        ));
    }
    if k < 2 || k > x.len() {
        return Err(format!(
            "ml_cross_validate: k must be in [2, {}], got {k}",
            x.len()
        ));
    }

    let params: std::collections::HashMap<String, f64> =
        if params_json.is_empty() || params_json.trim() == "null" {
            Default::default()
        } else {
            serde_json::from_str(params_json)
                .map_err(|e| format!("ml_cross_validate: bad params JSON: {e}"))?
        };

    let n = x.len();
    let mut folds = Vec::with_capacity(k);
    let mut mse_sum = 0.0f64;
    let mut r2_sum = 0.0f64;
    let mut scored = 0usize;

    for f in 0..k {
        let mut train_x: Vec<Vec<f64>> = Vec::new();
        let mut train_y: Vec<f64> = Vec::new();
        let mut val_x: Vec<Vec<f64>> = Vec::new();
        let mut val_y: Vec<f64> = Vec::new();
        for i in 0..n {
            if i % k == f {
                val_x.push(x[i].clone());
                val_y.push(y[i]);
            } else {
                train_x.push(x[i].clone());
                train_y.push(y[i]);
            }
        }
        let result = train::train(algorithm, &train_x, &train_y, &params)
            .map_err(|e| format!("ml_cross_validate: fold {f} training failed: {e}"))?;

        let (fold_mse, fold_r2) = match (result.mse, result.r_squared) {
            (Some(mse), Some(r2)) => {
                scored += 1;
                mse_sum += mse;
                r2_sum += r2;
                (Some(mse), Some(r2))
            }
            (Some(mse), None) => {
                scored += 1;
                mse_sum += mse;
                (Some(mse), None)
            }
            (None, Some(r2)) => {
                scored += 1;
                r2_sum += r2;
                (None, Some(r2))
            }
            (None, None) => (None, None),
        };

        let _ = val_x.len();
        folds.push(json!({
            "fold": f,
            "samples": val_x.len(),
            "mse": fold_mse,
            "r2": fold_r2,
        }));
    }

    let mean_mse = (scored > 0).then(|| mse_sum / scored as f64);
    let mean_r2 = (scored > 0).then(|| r2_sum / scored as f64);

    Ok(json!({
        "algorithm": algorithm.to_string(),
        "k": k,
        "folds": folds,
        "mean_mse": mean_mse,
        "mean_r2": mean_r2,
    })
    .to_string())
}

fn col_str(input: &RecordBatch, idx: usize) -> Result<&StringArray, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| format!("ml_cross_validate: arg {idx} is not VARCHAR").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn cv_regression_folds() {
        // y = 2x + 1, noiseless: every fold should be perfectly predictable.
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| 2.0 * i as f64 + 1.0).collect();
        let out: Value = serde_json::from_str(
            &cross_validate_from_json(
                "linear_regression",
                &serde_json::to_string(&x).unwrap(),
                &serde_json::to_string(&y).unwrap(),
                "null",
                5,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(out["algorithm"], "linear_regression");
        assert_eq!(out["k"], 5);
        assert_eq!(out["folds"].as_array().unwrap().len(), 5);
        assert!(out["mean_mse"].as_f64().unwrap() < 1e-9);
        assert!((out["mean_r2"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cv_k_fold_partition_exhausts() {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let out: Value = serde_json::from_str(
            &cross_validate_from_json(
                "linear_regression",
                &serde_json::to_string(&x).unwrap(),
                &serde_json::to_string(&y).unwrap(),
                "{}",
                4,
            )
            .unwrap(),
        )
        .unwrap();
        let samples: Vec<usize> = out["folds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["samples"].as_u64().unwrap() as usize)
            .collect();
        assert_eq!(samples.iter().sum::<usize>(), 8);
        assert_eq!(samples, vec![2, 2, 2, 2]);
    }

    #[test]
    fn cv_errors() {
        assert!(cross_validate_from_json("bogus", "[[1.0]]", "[1.0]", "null", 2).is_err());
        assert!(
            cross_validate_from_json("linear_regression", "[[1.0]]", "[1.0,2.0]", "null", 2)
                .is_err()
        );
        assert!(
            cross_validate_from_json("linear_regression", "[[1.0]]", "[1.0]", "null", 1).is_err()
        );
        assert!(cross_validate_from_json("linear_regression", "[]", "[]", "null", 2).is_err());
        assert!(
            cross_validate_from_json("linear_regression", "[[1.0]]", "[1.0]", "{bad", 2).is_err()
        );
    }

    #[test]
    fn cv_deterministic() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| (i as f64).sin()).collect();
        let a = cross_validate_from_json(
            "linear_regression",
            &serde_json::to_string(&x).unwrap(),
            &serde_json::to_string(&y).unwrap(),
            "null",
            3,
        )
        .unwrap();
        let b = cross_validate_from_json(
            "linear_regression",
            &serde_json::to_string(&x).unwrap(),
            &serde_json::to_string(&y).unwrap(),
            "null",
            3,
        )
        .unwrap();
        assert_eq!(a, b);
    }
}
