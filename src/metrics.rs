//! Prediction-quality metrics (MADlib `pred_metrics` counterpart).
//!
//! SQL surface:
//!
//! ```sql
//! -- binary (labels or probabilities; probabilities are thresholded at 0.5,
//! --        roc_auc uses the raw probabilities)
//! SELECT ml_metrics('[1, 0, 1]', '[0.9, 0.2, 0.7]', 'binary');
//! -- regression
//! SELECT ml_metrics('[1.5, 2.0]', '[1.4, 2.1]', 'regression');
//! -- auto: task inferred from the actual values ({0,1} → binary)
//! SELECT ml_metrics('[1, 0]', '[1, 0]');
//! ```
//!
//! Returns JSON. Binary: `confusion_matrix`, `accuracy`, `precision`,
//! `recall`, `f1`, `roc_auc`. Regression: `mse`, `rmse`, `mae`, `r2`.

use std::error::Error;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use duckdb::vscalar::arrow::{ArrowFunctionSignature, VArrowScalar};
use serde_json::{json, Value};

/// `ml_metrics(actual_json, predicted_json [, task])`
pub struct MetricsFn;

impl VArrowScalar for MetricsFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let n_args = input.num_columns();
        if !(2..=3).contains(&n_args) {
            return Err(
                "ml_metrics requires 2..3 args: actual_json, predicted_json[, task]".into(),
            );
        }

        let actual_col = col_str(&input, 0)?;
        let predicted_col = col_str(&input, 1)?;
        let task_col: Option<String> = if n_args >= 3 {
            let c = col_str(&input, 2)?;
            if c.is_null(0) {
                None
            } else {
                Some(c.value(0).to_lowercase())
            }
        } else {
            None
        };

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if actual_col.is_null(i) || predicted_col.is_null(i) {
                out.push(None);
                continue;
            }
            let task = task_col.as_deref().unwrap_or("auto");
            let json = metrics_from_json(actual_col.value(i), predicted_col.value(i), task)?;
            out.push(Some(json));
        }
        let arr = StringArray::from(out);
        Ok(Arc::new(arr))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![
            ArrowFunctionSignature::exact(
                vec![DataType::Utf8, DataType::Utf8, DataType::Utf8],
                DataType::Utf8,
            ),
            ArrowFunctionSignature::exact(vec![DataType::Utf8, DataType::Utf8], DataType::Utf8),
        ]
    }
}

/// Compute metrics over JSON number arrays. `task` ∈ {auto, binary, regression}.
pub fn metrics_from_json(
    actual_json: &str,
    predicted_json: &str,
    task: &str,
) -> Result<String, String> {
    let actual: Value = serde_json::from_str(actual_json)
        .map_err(|e| format!("ml_metrics: bad actual array: {e}"))?;
    let predicted: Value = serde_json::from_str(predicted_json)
        .map_err(|e| format!("ml_metrics: bad predicted array: {e}"))?;

    let a: Vec<f64> = to_f64_array(&actual, "actual")?;
    let p: Vec<f64> = to_f64_array(&predicted, "predicted")?;
    if a.is_empty() {
        return Err("ml_metrics: empty arrays".into());
    }
    if a.len() != p.len() {
        return Err(format!(
            "ml_metrics: length mismatch: actual {}, predicted {}",
            a.len(),
            p.len()
        ));
    }

    let is_binary =
        task == "binary" || (task == "auto" && a.iter().all(|v| *v == 0.0 || *v == 1.0));
    if task == "regression" {
        Ok(regression_json(&a, &p))
    } else if is_binary {
        Ok(binary_json(&a, &p))
    } else if task == "auto" {
        Ok(regression_json(&a, &p))
    } else {
        Err(format!(
            "ml_metrics: unknown task '{task}' (auto|binary|regression)"
        ))
    }
}

fn to_f64_array(v: &Value, name: &str) -> Result<Vec<f64>, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| format!("ml_metrics: {name} must be a JSON array"))?;
    arr.iter()
        .enumerate()
        .map(|(i, x)| {
            x.as_f64()
                .ok_or_else(|| format!("ml_metrics: {name}[{i}] is not a number"))
        })
        .collect()
}

/// Binary classification metrics. `predicted` may hold probabilities (any
/// float): they are thresholded at 0.5 for the confusion matrix, while ROC
/// AUC uses the raw scores.
fn binary_json(actual: &[f64], predicted: &[f64]) -> String {
    let n = actual.len();
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;
    for i in 0..n {
        let y = actual[i] > 0.5;
        let pred = predicted[i] > 0.5;
        match (y, pred) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
    }
    let accuracy = (tp + tn) as f64 / n as f64;
    let precision = safe_ratio(tp, tp + fp);
    let recall = safe_ratio(tp, tp + fn_);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let auc = roc_auc(actual, predicted);
    json!({
        "task": "binary",
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn_},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    })
    .to_string()
}

/// Area under the ROC curve via sorting by score (trapezoid rule).
fn roc_auc(actual: &[f64], scores: &[f64]) -> f64 {
    let mut pairs: Vec<(f64, f64)> = actual.iter().zip(scores).map(|(a, s)| (*a, *s)).collect();
    pairs.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = actual.iter().filter(|a| **a > 0.5).count();
    let n_neg = actual.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }

    // Rank-based: AUC = 1 - (Σ positive ranks - n_pos(n_pos+1)/2) / (n_pos·n_neg),
    // with ranks assigned on descending scores (rank 1 = highest score),
    // ties get the average rank.
    let mut rank_sum = 0.0f64;
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i;
        while j + 1 < pairs.len() && pairs[j + 1].1 == pairs[i].1 {
            j += 1;
        }
        // average rank for ties in [i, j], ranks are 1-based
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        let positives_in_tie = pairs[i..=j].iter().filter(|p| p.0 > 0.5).count();
        rank_sum += positives_in_tie as f64 * avg_rank;
        i = j + 1;
    }

    1.0 - (rank_sum - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0) / (n_pos as f64 * n_neg as f64)
}

fn regression_json(actual: &[f64], predicted: &[f64]) -> String {
    let n = actual.len();
    let mut mse = 0.0f64;
    let mut mae = 0.0f64;
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    let mean_y = actual.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        let e = actual[i] - predicted[i];
        mse += e * e;
        mae += e.abs();
        ss_res += e * e;
        let d = actual[i] - mean_y;
        ss_tot += d * d;
    }
    mse /= n as f64;
    mae /= n as f64;
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else if ss_res == 0.0 {
        1.0
    } else {
        f64::NEG_INFINITY
    };
    json!({
        "task": "regression",
        "mse": mse,
        "rmse": mse.sqrt(),
        "mae": mae,
        "r2": r2,
    })
    .to_string()
}

fn safe_ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

// ── arrow column helper (same shape as embed.rs / assoc_rules.rs) ──

fn col_str(input: &RecordBatch, idx: usize) -> Result<&StringArray, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| format!("ml_metrics: arg {idx} is not VARCHAR").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_perfect() {
        let out: Value =
            serde_json::from_str(&metrics_from_json("[1,0,1,0]", "[1,0,1,0]", "binary").unwrap())
                .unwrap();
        assert_eq!(out["task"], "binary");
        assert_eq!(out["confusion_matrix"]["tp"], 2);
        assert_eq!(out["confusion_matrix"]["tn"], 2);
        assert_eq!(out["confusion_matrix"]["fp"], 0);
        assert_eq!(out["confusion_matrix"]["fn"], 0);
        assert_eq!(out["accuracy"], 1.0);
        assert_eq!(out["precision"], 1.0);
        assert_eq!(out["recall"], 1.0);
        assert_eq!(out["f1"], 1.0);
        assert_eq!(out["roc_auc"], 1.0);
    }

    #[test]
    fn binary_imperfect_probs() {
        // probabilities: threshold at 0.5 → [1,0,0] vs actual [1,1,0]
        let out: Value =
            serde_json::from_str(&metrics_from_json("[1,1,0]", "[0.9,0.4,0.1]", "binary").unwrap())
                .unwrap();
        assert_eq!(out["confusion_matrix"]["tp"], 1);
        assert_eq!(out["confusion_matrix"]["fp"], 0);
        assert_eq!(out["confusion_matrix"]["tn"], 1);
        assert_eq!(out["confusion_matrix"]["fn"], 1);
        assert_eq!(out["recall"], 0.5);
        // scores [0.9, 0.4] > [0.1]: perfectly separated → AUC = 1.0
        assert_eq!(out["roc_auc"], 1.0);
    }

    #[test]
    fn regression_metrics() {
        // actual [1,2,3], pred [1,2,5]: e=[0,0,-2]
        let out: Value =
            serde_json::from_str(&metrics_from_json("[1,2,3]", "[1,2,5]", "regression").unwrap())
                .unwrap();
        assert!((out["mse"].as_f64().unwrap() - 4.0 / 3.0).abs() < 1e-9);
        assert!((out["rmse"].as_f64().unwrap() - (4.0f64 / 3.0).sqrt()).abs() < 1e-9);
        assert!((out["mae"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-9);
        // ss_tot = (1-2)²+(2-2)²+(3-2)² = 2; ss_res = 4; r2 = 1 - 4/2 = -1
        assert!((out["r2"].as_f64().unwrap() + 1.0).abs() < 1e-9);
    }

    #[test]
    fn auto_detects_binary() {
        let out: Value =
            serde_json::from_str(&metrics_from_json("[1,0,1]", "[1,1,0]", "auto").unwrap())
                .unwrap();
        assert_eq!(out["task"], "binary");
    }

    #[test]
    fn errors() {
        assert!(metrics_from_json("[1,0]", "[1]", "binary").is_err());
        assert!(metrics_from_json("[]", "[]", "binary").is_err());
        assert!(metrics_from_json("not json", "[1]", "binary").is_err());
        assert!(metrics_from_json("[1,0]", "[1,0]", "bogus").is_err());
    }

    #[test]
    fn auc_ties() {
        // ties in scores: [1,0,1] vs [0.5,0.5,0.5]
        let out: Value =
            serde_json::from_str(&metrics_from_json("[1,0,1]", "[0.5,0.5,0.5]", "binary").unwrap())
                .unwrap();
        assert_eq!(out["roc_auc"], 0.5);
    }
}
