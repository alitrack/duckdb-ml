//! Pure-Rust Gradient Boosting Decision Tree trainer
//!
//! Implements a simple but effective GBDT for regression.
//! Each tree fits the residuals from the previous ensemble.
//! Serializes to XGBoost-compatible JSON for ml_load_xgboost interop.

use super::tree::{build_tree, predict_tree, TreeNode, TreeParams};

/// One tree in the ensemble
#[derive(Debug, Clone)]
pub(crate) struct GbTree {
    pub tree: TreeNode,
    #[allow(dead_code)]
    pub base_score: f64,
}

/// GBDT training parameters
#[derive(Debug, Clone)]
pub struct GbdtParams {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub subsample: f64, // 0.0-1.0, 1.0 = no subsampling
}

impl Default for GbdtParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 6,
            min_samples_split: 5,
            subsample: 1.0,
        }
    }
}

/// GBDT ensemble
#[derive(Debug, Clone)]
pub struct GbdtEnsemble {
    pub(crate) trees: Vec<GbTree>,
    pub initial_prediction: f64,
    pub n_features: usize,
    pub n_samples: usize,
    pub params: GbdtParams,
    pub objective: GbdtObjective,
}

/// GBDT training objective
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbdtObjective {
    /// Squared error — regression on raw targets
    SquaredError,
    /// Logistic (binary:logistic) — trees fit Newton steps of log-loss
    Logistic,
}

/// Sigmoid link, overflow-safe
fn sigmoid(z: f64) -> f64 {
    if z >= 30.0 {
        1.0
    } else if z <= -30.0 {
        0.0
    } else {
        1.0 / (1.0 + (-z).exp())
    }
}

/// Rewrite every leaf of a freshly built tree with the Newton step
/// Σ(g·h)/Σh of log-loss — this is exactly what sklearn's
/// GradientBoostingClassifier does (MSE splits on the gradient, Newton
/// leaves). `g`/`h` are the arrays the tree was built on (already
/// subsampled if applicable).
fn apply_newton_leaves(tree: &mut TreeNode, x: &[Vec<f64>], g: &[f64], h: &[f64], indices: &[usize]) {
    match tree {
        TreeNode::Leaf { value } => {
            let mut gs = 0.0f64;
            let mut hs = 0.0f64;
            for &i in indices {
                gs += g[i];
                hs += h[i];
            }
            *value = if hs.abs() < 1e-12 {
                0.0
            } else {
                // Newton step fits the NEGATIVE gradient: g = p − y is the
                // log-loss gradient, so the leaf is −Σg/Σh (xgboost/sklearn
                // convention — fitting +Σg/Σh inverts the classes).
                -gs / hs
            };
        }
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            let mut li = Vec::with_capacity(indices.len());
            let mut ri = Vec::with_capacity(indices.len());
            for &i in indices {
                if x[i][*feature_index] <= *threshold {
                    li.push(i);
                } else {
                    ri.push(i);
                }
            }
            apply_newton_leaves(left, x, g, h, &li);
            apply_newton_leaves(right, x, g, h, &ri);
        }
    }
}

/// Train a GBDT ensemble for regression (squared error loss)
pub fn train_gbdt(
    x: &[Vec<f64>],
    y: &[f64],
    params: &GbdtParams,
    objective: GbdtObjective,
) -> GbdtEnsemble {
    let n_samples = x.len();
    let n_features = x[0].len();
    // Squared error starts from the mean; logistic starts from raw 0
    // (p = 0.5), matching sklearn's init='zero' / xgboost base_score=0.5.
    let initial_prediction = match objective {
        GbdtObjective::SquaredError => y.iter().sum::<f64>() / n_samples as f64,
        GbdtObjective::Logistic => 0.0,
    };

    // Current predictions — start with mean
    let mut predictions = vec![initial_prediction; n_samples];
    let mut trees = Vec::with_capacity(params.n_estimators);

    for _iter in 0..params.n_estimators {
        // Tree targets: squared error → residuals; logistic → log-loss
        // gradient g = p − y (with p = sigmoid(raw)) and hessian h = p(1−p).
        let mut targets: Vec<f64> = Vec::with_capacity(n_samples);
        let mut hessian: Option<Vec<f64>> = None;
        match objective {
            GbdtObjective::SquaredError => {
                targets = y
                    .iter()
                    .zip(predictions.iter())
                    .map(|(&yi, &pi)| yi - pi)
                    .collect();
            }
            GbdtObjective::Logistic => {
                let mut h = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let p = sigmoid(predictions[i]);
                    targets.push(p - y[i]);
                    h.push((p * (1.0 - p)).max(1e-12));
                }
                hessian = Some(h);
            }
        }

        // Subsample if requested
        let (x_sub, targets_sub, h_sub): (Vec<Vec<f64>>, Vec<f64>, Option<Vec<f64>>) =
            if params.subsample < 1.0 {
                let n_sub = (n_samples as f64 * params.subsample) as usize;
                let n_sub = n_sub.max(2);
                let _indices: Vec<usize> = (0..n_samples).collect();
                // Simple deterministic sampling (every k-th row)
                let step = (n_samples as f64 / n_sub as f64).ceil() as usize;
                let selected: Vec<usize> = (0..n_samples).step_by(step).take(n_sub).collect();
                let xs: Vec<Vec<f64>> = selected.iter().map(|&i| x[i].clone()).collect();
                let ts: Vec<f64> = selected.iter().map(|&i| targets[i]).collect();
                let hs = hessian.as_ref().map(|h| selected.iter().map(|&i| h[i]).collect());
                (xs, ts, hs)
            } else {
                (x.to_vec(), targets, hessian)
            };

        let tp = TreeParams {
            max_depth: params.max_depth,
            min_samples_split: params.min_samples_split,
            min_samples_leaf: 1,
            max_features: None,
        };

        let mut tree = build_tree(&x_sub, &targets_sub, &tp);

        // Logistic: replace leaves with Newton steps Σg/Σh
        if objective == GbdtObjective::Logistic {
            let h = h_sub.as_ref().expect("hessian present for logistic");
            let all: Vec<usize> = (0..x_sub.len()).collect();
            apply_newton_leaves(&mut tree, &x_sub, &targets_sub, h, &all);
        }

        // Update predictions
        for i in 0..n_samples {
            let update = params.learning_rate * predict_tree(&tree, &x[i]);
            predictions[i] += update;
        }

        trees.push(GbTree {
            tree,
            base_score: 0.0,
        });
    }

    GbdtEnsemble {
        trees,
        initial_prediction,
        n_features,
        n_samples,
        params: params.clone(),
        objective,
    }
}

impl GbdtEnsemble {
    /// Link the raw ensemble score: identity for squared error, sigmoid
    /// for logistic (probability scale).
    pub fn link(&self, raw: f64) -> f64 {
        match self.objective {
            GbdtObjective::SquaredError => raw,
            GbdtObjective::Logistic => sigmoid(raw),
        }
    }

    /// Predict for a single sample (raw score)
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut pred = self.initial_prediction;
        for gbt in &self.trees {
            pred += self.params.learning_rate * predict_tree(&gbt.tree, features);
        }
        pred
    }

    /// Predict probability (logistic only; identity for regression)
    pub fn predict_prob(&self, features: &[f64]) -> f64 {
        self.link(self.predict(features))
    }

    /// Compute R-squared
    pub fn r_squared(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        if ss_tot == 0.0 {
            return 0.0;
        }
        let ss_res: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, &yi)| {
                let pred = self.predict_prob(xi);
                (yi - pred).powi(2)
            })
            .sum();
        1.0 - ss_res / ss_tot
    }

    /// Compute MSE
    pub fn mse(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let n = y.len() as f64;
        let sum: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, &yi)| {
                let pred = self.predict_prob(xi);
                (yi - pred).powi(2)
            })
            .sum();
        sum / n
    }

    /// Serialize to XGBoost-compatible JSON format
    /// Serialize to XGBoost-compatible JSON.
    ///
    /// Faithful to XGBoost semantics so that `ml_load_xgboost`-style parsing
    /// reproduces the training-time ensemble exactly:
    ///   - `base_score` = the training-time initial prediction (mean of y),
    ///     NOT a hardcoded 0.5 — a wrong base shifts every prediction
    ///   - leaf weights are pre-multiplied by `learning_rate` (XGBoost bakes
    ///     eta into leaves at training time; the JSON predictor sums leaves
    ///     without re-applying it)
    ///   - objective is task-dependent: `reg:squarederror` for regression,
    ///     `binary:logistic` for binary classification (drives sigmoid at
    ///     predict time)
    pub fn to_xgb_json(&self, objective: &str) -> String {
        let trees_json: Vec<String> = self
            .trees
            .iter()
            .enumerate()
            .map(|(idx, gbt)| serialize_tree_json(&gbt.tree, idx, self.params.learning_rate))
            .collect();

        let tree_info: Vec<String> = (0..self.trees.len()).map(|_| "0".to_string()).collect();

        format!(
            r#"{{"version":[2,0,0],"learner":{{"gradient_booster":{{"name":"gbtree","model":{{"gbtree_model_param":{{"num_trees":"{trees_len}","num_features":"{n_feat}"}},"trees":[{trees}],"tree_info":[{tinfo}]}}}},"learner_model_param":{{"base_score":"{base}","num_class":"0","num_feature":"{n_feat}"}},"objective":{{"name":"{objective}","reg_loss_param":{{"scale_pos_weight":"1"}}}},"attributes":{{"scikit_learn":{{"n_estimators":{n_est},"max_depth":{md},"learning_rate":{lr}}}}}}}}}"#,
            trees_len = self.trees.len(),
            n_feat = self.n_features,
            tinfo = tree_info.join(","),
            trees = trees_json.join(","),
            base = format!("{:.12E}", self.initial_prediction),
            objective = objective,
            n_est = self.params.n_estimators,
            md = self.params.max_depth,
            lr = self.params.learning_rate,
        )
    }
}

/// Serialize a single tree to XGBoost JSON format.
///
/// `learning_rate` scales leaf weights so the JSON predictor
/// (`base_score + Σ leaf`) matches the training-time update
/// (`pred += lr · tree(x)`), exactly like XGBoost's eta.
fn serialize_tree_json(tree: &TreeNode, tree_id: usize, learning_rate: f64) -> String {
    let mut nodes = Vec::new();
    let mut stats = Vec::new();
    serialize_node(tree, &mut nodes, &mut stats, 0, learning_rate);
    let n_nodes = nodes.len();

    // Build arrays
    let mut left_children = vec![-1i32; n_nodes];
    let mut right_children = vec![-1i32; n_nodes];
    let mut parents = vec![2147483647i32; n_nodes]; // max i32 = missing parent
    let mut split_indices = vec![0u32; n_nodes];
    let mut split_conditions = vec![0.0f64; n_nodes];
    let default_left = vec![false; n_nodes];
    let mut base_weights = vec![0.0f64; n_nodes];

    // Root parent stays as max i32
    for (i, n) in nodes.iter().enumerate() {
        match n {
            SerNode::Split {
                feat,
                thresh,
                left_idx,
                right_idx,
            } => {
                split_indices[i] = *feat as u32;
                split_conditions[i] = *thresh;
                left_children[i] = *left_idx as i32;
                right_children[i] = *right_idx as i32;
                if *left_idx > 0 {
                    parents[*left_idx] = i as i32;
                }
                if *right_idx > 0 {
                    parents[*right_idx] = i as i32;
                }
            }
            SerNode::Leaf { weight } => {
                base_weights[i] = *weight;
            }
        }
    }

    format!(
        r#"{{"base_weights":{bw},"categories":[],"categories_nodes":[],"categories_segments":[],"categories_sizes":[],"default_left":{dl},"id":{tid},"left_children":{lc},"loss_changes":{ls},"parents":{ps},"right_children":{rc},"split_conditions":{sc},"split_indices":{si},"split_type":{st},"sum_hessian":{sh},"tree_param":{{"num_deleted":"0","num_feature":"{nf}","num_nodes":"{nn}","size_leaf_vector":"0"}}}}"#,
        bw = format_f64_array(&base_weights),
        dl = format_bool_array(&default_left),
        tid = tree_id,
        lc = format_i32_array(&left_children),
        ls = format_f64_array(&vec![0.0f64; n_nodes]),
        ps = format_i32_array(&parents),
        rc = format_i32_array(&right_children),
        sc = format_f64_array(&split_conditions),
        si = format_u32_array(&split_indices),
        st = format_i32_array(&vec![0i32; n_nodes]),
        sh = format_f64_array(&vec![0.0f64; n_nodes]),
        nf = split_indices.len().max(1),
        nn = n_nodes,
    )
}

enum SerNode {
    Split {
        feat: usize,
        thresh: f64,
        left_idx: usize,
        right_idx: usize,
    },
    Leaf {
        weight: f64,
    },
}

fn serialize_node(
    tree: &TreeNode,
    nodes: &mut Vec<SerNode>,
    _stats: &mut Vec<f64>,
    _depth: usize,
    learning_rate: f64,
) -> usize {
    let idx = nodes.len();
    match tree {
        TreeNode::Leaf { value } => {
            nodes.push(SerNode::Leaf {
                weight: *value * learning_rate,
            });
        }
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            nodes.push(SerNode::Split {
                feat: *feature_index,
                thresh: *threshold,
                left_idx: 0, // placeholder, will update
                right_idx: 0,
            });
            let li = serialize_node(left, nodes, _stats, _depth + 1, learning_rate);
            let ri = serialize_node(right, nodes, _stats, _depth + 1, learning_rate);
            if let SerNode::Split {
                ref mut left_idx,
                ref mut right_idx,
                ..
            } = nodes[idx]
            {
                *left_idx = li;
                *right_idx = ri;
            }
        }
    }
    idx
}

fn format_f64_array(v: &[f64]) -> String {
    let items: Vec<String> = v
        .iter()
        .map(|x| {
            let s = format!("{:.6}", x)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string();
            if s.is_empty() || s == "-" {
                "0.0".to_string()
            } else {
                s
            }
        })
        .collect();
    format!("[{}]", items.join(","))
}

fn format_i32_array(v: &[i32]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(","))
}

fn format_u32_array(v: &[u32]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(","))
}

fn format_bool_array(v: &[bool]) -> String {
    let items: Vec<String> = v
        .iter()
        .map(|x| {
            if *x {
                "true".to_string()
            } else {
                "false".to_string()
            }
        })
        .collect();
    format!("[{}]", items.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbdt_simple_regression() {
        // y = 3*x1 + 2*x2 + noise
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
            vec![6.0, 5.0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 2.0 * xi[1] + 1.0).collect();

        let params = GbdtParams {
            n_estimators: 5,
            learning_rate: 0.3,
            max_depth: 2,
            ..Default::default()
        };

        let ensemble = train_gbdt(&x, &y, &params, GbdtObjective::SquaredError);
        let r2 = ensemble.r_squared(&x, &y);
        assert!(r2 > 0.7, "GBDT R² too low: {r2}");

        let pred = ensemble.predict(&[3.0, 3.0]);
        assert!(pred.is_finite(), "pred not finite: {pred}");

        // Test serialization produces valid XGBoost JSON
        let json = ensemble.to_xgb_json("reg:squarederror");
        assert!(json.contains("\"gbtree\""));
        assert!(json.contains("\"trees\""));

        // Verify JSON is parseable — print error position for debugging
        match serde_json::from_str::<serde_json::Value>(&json) {
            Ok(_) => {}
            Err(e) => {
                let col = e.column();
                let start = if col > 80 { col - 80 } else { 0 };
                let end = (col + 80).min(json.len());
                panic!("JSON error at col {col}: '{}...'", &json[start..end]);
            }
        }
    }

    #[test]
    fn test_gbdt_json_roundtrip_matches_ensemble() {
        // THE layer the old test missed: training-time predictions go through
        // to_xgb_json -> XgbModel::from_json -> predict in production
        // (ml_train stores the JSON blob). The JSON predictor must reproduce
        // the in-process ensemble exactly.
        let x = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 4.0],
            vec![4.0, 3.0],
            vec![5.0, 6.0],
            vec![6.0, 5.0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] + 2.0 * xi[1] + 1.0).collect();
        let params = GbdtParams {
            n_estimators: 8,
            learning_rate: 0.2,
            max_depth: 3,
            ..Default::default()
        };
        let ensemble = train_gbdt(&x, &y, &params, GbdtObjective::SquaredError);

        // regression objective
        let json = ensemble.to_xgb_json("reg:squarederror");
        assert!(json.contains("reg:squarederror"));
        assert!(json.contains(&format!("{:.12E}", ensemble.initial_prediction)),
            "base_score must be the training mean, got {}", json);
        let model = crate::model::xgboost::XgbModel::from_json(json.as_bytes())
            .expect("JSON must parse");
        for xi in &x {
            let a = ensemble.predict(xi);
            let b = model.predict(xi).expect("predict");
            assert!(
                (a - b).abs() < 1e-6,
                "roundtrip mismatch: ensemble={a} json={b} for {xi:?}"
            );
        }

        // binary objective must serialize as binary:logistic (drives sigmoid)
        let json_bin = ensemble.to_xgb_json("binary:logistic");
        assert!(json_bin.contains("binary:logistic"));
        let model_bin = crate::model::xgboost::XgbModel::from_json(json_bin.as_bytes())
            .expect("binary JSON must parse");
        for xi in &x {
            let raw = model_bin.predict_raw(xi).expect("raw");
            let prob = model_bin.predict(xi).expect("predict");
            let sig = 1.0 / (1.0 + (-raw).exp());
            assert!(
                (prob - sig).abs() < 1e-9,
                "binary predict must apply sigmoid: {prob} vs {sig}"
            );
        }
    }

    #[test]
    fn test_gbdt_logistic_separates_blobs() {
        // THE bug labs caught: xgboost_binary trained 0/1 targets with
        // squared error AND applied sigmoid at predict → double transform
        // collapsed every prediction into [0.5, ~0.73] (random accuracy).
        // With the logistic objective (Newton leaves, raw scale = logit),
        // sigmoid(raw) must separate the classes again.
        let rng = |seed: u64| {
            let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            move || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0 // ~[-1, 1]
            }
        };
        let mut r = rng(42);
        let mut x = Vec::new();
        let mut y = Vec::new();
        // two blobs centered ±1.5 (mix a little so it isn't linearly trivial)
        for i in 0..200 {
            let side = if i < 100 { -1.5 } else { 1.5 };
            x.push(vec![side + r() * 0.7, side + r() * 0.7]);
            y.push(if i < 100 { 0.0 } else { 1.0 });
        }
        let params = GbdtParams {
            n_estimators: 80,
            learning_rate: 0.1,
            max_depth: 4,
            ..Default::default()
        };
        let ensemble = train_gbdt(&x, &y, &params, GbdtObjective::Logistic);

        // roundtrip through the JSON path, then evaluate on the raw scale
        let json = ensemble.to_xgb_json("binary:logistic");
        let model = crate::model::xgboost::XgbModel::from_json(json.as_bytes())
            .expect("binary JSON must parse");
        let mut correct = 0usize;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let prob = model.predict(xi).expect("predict");
            let pred_class = if prob >= 0.5 { 1.0 } else { 0.0 };
            if (pred_class - yi).abs() < 1e-12 {
                correct += 1;
            }
        }
        let acc = correct as f64 / x.len() as f64;
        assert!(acc > 0.95, "logistic GBDT accuracy too low: {acc}");
    }
}
