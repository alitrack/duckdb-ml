//! Pure-Rust XGBoost model parser and inference engine
//!
//! Parses XGBoost JSON model format (columnar tree layout),
//! performs prediction without libxgboost.
//!
//! Supported: gbtree booster, reg:squarederror and reg:logistic objectives.

use serde_json::Value;

/// A single tree in columnar format
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct XgbTree {
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<i32>,
    split_conditions: Vec<f64>,
    base_weights: Vec<f64>,
    default_left: Vec<u8>, // reserved for future missing-value handling
}

/// Parsed XGBoost model
#[derive(Debug, Clone)]
pub struct XgbModel {
    base_score: f64,
    num_features: usize,
    trees: Vec<XgbTree>,
    objective: String,
}

impl XgbTree {
    fn predict(&self, features: &[f64]) -> f64 {
        let mut nid = 0i32;
        loop {
            if self.left_children[nid as usize] == -1 && self.right_children[nid as usize] == -1 {
                return self.base_weights[nid as usize];
            }
            let f = self.split_indices[nid as usize] as usize;
            let t = self.split_conditions[nid as usize];
            if features.get(f).copied().unwrap_or(f64::NAN) < t {
                nid = self.left_children[nid as usize];
            } else {
                nid = self.right_children[nid as usize];
            }
        }
    }
}

impl XgbModel {
    /// Parse XGBoost JSON model from bytes
    pub fn from_json(json_bytes: &[u8]) -> Result<Self, String> {
        let root: Value =
            serde_json::from_slice(json_bytes).map_err(|e| format!("JSON parse error: {e}"))?;

        let learner = root
            .get("learner")
            .ok_or("missing 'learner' key in XGBoost model")?;

        let param = learner
            .get("learner_model_param")
            .ok_or("missing learner_model_param")?;

        let base_score = parse_val(param.get("base_score").ok_or("missing base_score")?);
        let num_features = param
            .get("num_feature")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        let objective = learner
            .get("objective")
            .and_then(|o| o.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("reg:squarederror")
            .to_string();

        let gb = learner
            .get("gradient_booster")
            .ok_or("missing gradient_booster")?;

        let model_block = gb.get("model").ok_or("missing model block")?;
        let tree_list = model_block
            .get("trees")
            .ok_or("missing trees")?
            .as_array()
            .ok_or("trees is not an array")?;

        let mut trees = Vec::with_capacity(tree_list.len());

        for tree_json in tree_list {
            let left_children = parse_i32_array(tree_json, "left_children")?;
            let right_children = parse_i32_array(tree_json, "right_children")?;
            let split_indices = parse_i32_array(tree_json, "split_indices")?;
            let split_conditions = parse_f64_array(tree_json, "split_conditions")?;
            let base_weights = parse_f64_array(tree_json, "base_weights")?;
            let default_left: Vec<u8> = tree_json
                .get("default_left")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().map(|x| x.as_i64().unwrap_or(0) as u8).collect())
                .unwrap_or_else(|| vec![0u8; left_children.len()]);

            let nn = left_children.len();
            if right_children.len() != nn
                || split_indices.len() != nn
                || split_conditions.len() != nn
                || base_weights.len() != nn
            {
                return Err("Tree arrays have inconsistent lengths".into());
            }

            trees.push(XgbTree {
                left_children,
                right_children,
                split_indices,
                split_conditions,
                base_weights,
                default_left,
            });
        }

        Ok(XgbModel {
            base_score,
            num_features,
            trees,
            objective,
        })
    }

    /// Predict raw score before applying objective link
    pub fn predict_raw(&self, features: &[f64]) -> Result<f64, String> {
        if features.len() != self.num_features {
            return Err(format!(
                "feature count mismatch: expected {}, got {}",
                self.num_features,
                features.len()
            ));
        }
        let score: f64 =
            self.base_score + self.trees.iter().map(|t| t.predict(features)).sum::<f64>();
        Ok(score)
    }

    /// Predict with objective transform (regression: identity, binary:logistic: sigmoid)
    pub fn predict(&self, features: &[f64]) -> Result<f64, String> {
        let raw = self.predict_raw(features)?;
        match self.objective.as_str() {
            "reg:squarederror"
            | "reg:squaredlogerror"
            | "reg:absoluteerror"
            | "reg:pseudohubererror" => Ok(raw),
            "binary:logistic" | "reg:logistic" => Ok(1.0 / (1.0 + (-raw).exp())),
            other => Err(format!("unsupported objective: {other}")),
        }
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// Serialize this model to bytes (store the raw JSON)
    pub fn to_bytes(&self, json_data: &[u8]) -> Vec<u8> {
        json_data.to_vec()
    }
}

// ——— helpers ———

fn parse_val(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s
            .trim_matches(|c| c == '[' || c == ']')
            .parse()
            .unwrap_or(0.0),
        _ => 0.0,
    }
}

fn parse_i32_array(tree: &Value, key: &str) -> Result<Vec<i32>, String> {
    tree.get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing key '{key}' or not an array"))?
        .iter()
        .map(|v| {
            let s = match v {
                Value::Number(n) => n.as_i64().ok_or("not int".to_string()),
                Value::String(s) => s.parse::<i64>().map_err(|e| e.to_string()),
                _ => Err("not number/string".to_string()),
            }?;
            Ok(s as i32)
        })
        .collect()
}

fn parse_f64_array(tree: &Value, key: &str) -> Result<Vec<f64>, String> {
    tree.get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing key '{key}' or not an array"))?
        .iter()
        .map(|v| Ok(parse_val(v)))
        .collect()
}

// ——— MlModel integration ———

use super::{Algorithm, MlModel, ModelError, ModelMetadata};

/// XGBoost model wrapped for the MlModel trait.
/// Stores the raw JSON bytes for serialization.
pub struct XgbModelWrapper {
    model: XgbModel,
    json_bytes: Vec<u8>,
    metadata: ModelMetadata,
}

impl XgbModelWrapper {
    pub fn new(json_bytes: Vec<u8>) -> Result<Self, ModelError> {
        let model = XgbModel::from_json(&json_bytes).map_err(ModelError::Serialization)?;

        let is_classifier = model.objective.contains("logistic");
        let algo = if is_classifier {
            Algorithm::XGBoostClassifier
        } else {
            Algorithm::XGBoostRegressor
        };

        let metadata = ModelMetadata {
            algorithm: algo,
            num_features: model.num_features(),
            num_samples: 0,
            r_squared: None,
            mse: None,
            coefficients_count: model.n_trees(),
            hyperparameters_json: serde_json::json!({
                "n_trees": model.n_trees(),
                "objective": model.objective(),
            })
            .to_string(),
        };

        Ok(Self {
            model,
            json_bytes,
            metadata,
        })
    }
}

impl MlModel for XgbModelWrapper {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        self.model.predict(features).map_err(ModelError::Training)
    }

    fn algorithm(&self) -> Algorithm {
        self.metadata.algorithm
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(self.json_bytes.clone())
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        XgbModelWrapper::new(blob.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xgb_model_parse_and_predict() {
        // Small XGBoost JSON model (hand-crafted to match xgb format)
        let json = r#"{
            "version": [2, 1, 3],
            "learner": {
                "learner_model_param": {
                    "base_score": "5.0E0",
                    "num_feature": "2"
                },
                "objective": {"name": "reg:squarederror"},
                "gradient_booster": {
                    "name": "gbtree",
                    "model": {
                        "gbtree_model_param": {"num_trees": "2"},
                        "trees": [
                            {
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "split_indices": [0, 0, 0],
                                "split_conditions": [0.5, 0.0, 0.0],
                                "base_weights": [0.0, -1.0, 2.0],
                                "default_left": [0, 0, 0]
                            },
                            {
                                "left_children": [-1, -1],
                                "right_children": [-1, -1],
                                "split_indices": [0, 0],
                                "split_conditions": [0.0, 0.0],
                                "base_weights": [1.5, 1.5],
                                "default_left": [0, 0]
                            }
                        ]
                    }
                }
            }
        }"#;

        let model = XgbModel::from_json(json.as_bytes()).unwrap();
        assert_eq!(model.base_score, 5.0);
        assert_eq!(model.num_features, 2);
        assert_eq!(model.n_trees(), 2);

        // Tree 0: split on feature[0] < 0.5 → left: -1.0, right: 2.0, root: 0.0
        // Tree 1: leaf only: 1.5
        // Base: 5.0
        // For [0.0, 0.0]: 5.0 + (-1.0) + 1.5 = 5.5? No:
        //   Tree 0: 0.0 < 0.5 → left leaf = -1.0 ✓
        //   Tree 1: leaf = 1.5 ✓
        //   Total: 5.0 + (-1.0) + 1.5 = 5.5
        assert!((model.predict(&[0.0, 0.0]).unwrap() - 5.5).abs() < 0.01);

        // For [1.0, 1.0]: 5.0 + 2.0 + 1.5 = 8.5
        assert!((model.predict(&[1.0, 1.0]).unwrap() - 8.5).abs() < 0.01);
    }

    #[test]
    fn test_xgb_regression_roundtrip() {
        // Parse the real xgb model
        let json_bytes = std::fs::read("/tmp/xgb_model.json")
            .expect("run python script first to generate /tmp/xgb_model.json");

        let model = XgbModel::from_json(&json_bytes).unwrap();
        assert_eq!(model.num_features, 3);
        assert_eq!(model.n_trees(), 1);

        // Known predictions from the Python reference run
        let p0 = model.predict(&[1.0, 2.0, 3.0]).unwrap();
        assert!((p0 - 31.0).abs() < 0.5, "p0={p0}");

        let p2 = model.predict(&[7.0, 8.0, 9.0]).unwrap();
        assert!((p2 - 34.25).abs() < 0.5, "p2={p2}");
    }
}
