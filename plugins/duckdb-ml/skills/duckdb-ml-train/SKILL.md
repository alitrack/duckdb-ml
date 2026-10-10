---
name: duckdb-ml-train
description: >
  Train and run machine-learning models in pure SQL with the ml DuckDB
  extension (34 algorithms, zero Python): regression, SVM/SVR, trees and
  forests, logistic/multinomial/ordinal, survival (Cox, Kaplan-Meier),
  ARIMA. Use when a query needs prediction, scoring, or forecasting without
  leaving DuckDB or adding a Python ML stack.
version: 0.2.0
---

# ml: training & inference

Extension: `ml` (pure Rust, 34 algorithms, zero Python deps).
`INSTALL ml FROM community; LOAD ml;`
Features are JSON arrays (`'[3.0, 4.5]'` / `'[[1,2],[3,4]]'`), labels a
plain JSON array; hyperparameters a JSON object.

## Lifecycle

```sql
-- Train (name, algorithm, y, X, params)
SELECT * FROM ml_train('my_model', 'random_forest', '[..]', '[[..]]', '{"n_estimators":100}');

-- Predict single / batch
SELECT * FROM ml_predict('my_model', '[3.0, 4.5]');
SELECT * FROM ml_predict_batch('my_model', '[[1.0,2.0],[3.0,4.0]]');
SELECT * FROM ml_predict_batch('my_model', '@my_model');   -- re-use training data
```

## Algorithm picker

| Need | Algorithm string |
|---|---|
| Plain regression | `linear_regression` |
| Regularized | `ridge_regression`, `lasso_regression`, `elastic_net` |
| Outliers present | `robust` (Huber) |
| Nonlinear curve | `polynomial_regression` (degree ≥ 1, per-feature powers) |
| Classification | `svm`, `logistic_regression` (binary), `multilogistic` (softmax), `ordinal` (ordered classes) |
| Regression w/ kernels | `svr` (linear/rbf/poly/sigmoid, working-set SMO) |
| Trees / ensembles | `decision_tree`, `random_forest` (regression), `rf_classifier` (Gini, majority vote, string labels auto-encoded), `adaboost` (SAMME) |
| Survival | `ml_cox_train` (Cox PH), `ml_km_train` (Kaplan-Meier; predict = median survival) |
| Forecasting | `arima` (ARIMA(p,d,q)) |
| Class imbalance | `ml_smote(x_json, y_json, k, dup_ratio)` — deterministic oversampling |
| Blend models | `ml_voting(models_json, features_json, mode)` — hard majority / mean |

## Deployment & data versioning

```sql
SELECT * FROM ml_deploy('my_model', 'best_score');   -- promote best
SELECT * FROM ml_deploy('my_model', 'rollback');     -- undo last deploy

-- Snapshot training data lineage: name, label, n_rows?, n_features?, target, feature names, hash
SELECT * FROM ml_snapshot('my_model', 'train_data', 4, 250, 'target', '["x1","x2"]', 'abc123');
SELECT * FROM ml_list_snapshots('my_model');
```

## Gotchas

- Unsupervised algorithms accept empty target (`''` or `[]`).
- Model registry is per-database — reopen the same .db to reuse models.
- Scale features for SVM/SVR/regularized models; algorithms do not
  auto-scale.
