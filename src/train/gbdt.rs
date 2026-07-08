//! Pure-Rust Gradient Boosting Decision Tree trainer
//!
//! Implements a simple but effective GBDT for regression.
//! Each tree fits the residuals from the previous ensemble.
//! Serializes to XGBoost-compatible JSON for ml_load_xgboost interop.

use super::tree::{TreeNode, build_tree, TreeParams, predict_tree};

/// One tree in the ensemble
#[derive(Debug, Clone)]
struct GbTree {
    tree: TreeNode,
    base_score: f64,
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
    pub trees: Vec<GbTree>,
    pub initial_prediction: f64,
    pub n_features: usize,
    pub n_samples: usize,
    pub params: GbdtParams,
}

/// Train a GBDT ensemble for regression (squared error loss)
pub fn train_gbdt(
    x: &[Vec<f64>],
    y: &[f64],
    params: &GbdtParams,
) -> GbdtEnsemble {
    let n_samples = x.len();
    let n_features = x[0].len();
    let initial_prediction = y.iter().sum::<f64>() / n_samples as f64;

    // Current predictions — start with mean
    let mut predictions = vec![initial_prediction; n_samples];
    let mut trees = Vec::with_capacity(params.n_estimators);

    for _iter in 0..params.n_estimators {
        // Compute residuals (pseudo-residuals for squared error)
        let residuals: Vec<f64> = y.iter()
            .zip(predictions.iter())
            .map(|(&yi, &pi)| yi - pi)
            .collect();

        // Subsample if requested
        let (x_sub, residuals_sub): (Vec<Vec<f64>>, Vec<f64>) = if params.subsample < 1.0 {
            let n_sub = (n_samples as f64 * params.subsample) as usize;
            let n_sub = n_sub.max(2);
            let mut indices: Vec<usize> = (0..n_samples).collect();
            // Simple deterministic sampling (every k-th row)
            let step = (n_samples as f64 / n_sub as f64).ceil() as usize;
            let selected: Vec<usize> = (0..n_samples).step_by(step).take(n_sub).collect();
            let xs: Vec<Vec<f64>> = selected.iter().map(|&i| x[i].clone()).collect();
            let rs: Vec<f64> = selected.iter().map(|&i| residuals[i]).collect();
            (xs, rs)
        } else {
            (x.to_vec(), residuals)
        };

        let tp = TreeParams {
            max_depth: params.max_depth,
            min_samples_split: params.min_samples_split,
            min_samples_leaf: 1,
            max_features: None,
        };

        let tree = build_tree(&x_sub, &residuals_sub, &tp);

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
    }
}

impl GbdtEnsemble {
    /// Predict for a single sample
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut pred = self.initial_prediction;
        for gbt in &self.trees {
            pred += self.params.learning_rate * predict_tree(&gbt.tree, features);
        }
        pred
    }

    /// Compute R-squared
    pub fn r_squared(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        if ss_tot == 0.0 {
            return 0.0;
        }
        let ss_res: f64 = x.iter()
            .zip(y.iter())
            .map(|(xi, &yi)| {
                let pred = self.predict(xi);
                (yi - pred).powi(2)
            })
            .sum();
        1.0 - ss_res / ss_tot
    }

    /// Compute MSE
    pub fn mse(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let n = y.len() as f64;
        let sum: f64 = x.iter()
            .zip(y.iter())
            .map(|(xi, &yi)| {
                let pred = self.predict(xi);
                (yi - pred).powi(2)
            })
            .sum();
        sum / n
    }

    /// Serialize to XGBoost-compatible JSON format
    pub fn to_xgb_json(&self) -> String {
        let trees_json: Vec<String> = self.trees.iter().enumerate().map(|(idx, gbt)| {
            let tree_json = serialize_tree_json(&gbt.tree, idx);
            tree_json
        }).collect();

        format!(
            r#"{{"version":[2,0,0],"learner":{{"gradient_booster":{{"name":"gbtree","model":{{"gbtree_model_param":{{"num_trees":"{}","num_features":"{}"}},"trees":[{trees}]}}}},"attributes":{{"scikit_learn":{{"n_estimators":{n_est},"max_depth":{md},"learning_rate":{lr}}}}}}}}}"#,
            self.trees.len(),
            self.n_features,
            trees = trees_json.join(","),
            n_est = self.params.n_estimators,
            md = self.params.max_depth,
            lr = self.params.learning_rate,
        )
    }
}

/// Serialize a single tree to XGBoost JSON format
fn serialize_tree_json(tree: &TreeNode, tree_id: usize) -> String {
    let mut nodes = Vec::new();
    let mut stats = Vec::new();
    serialize_node(tree, &mut nodes, &mut stats, 0);
    let n_nodes = nodes.len();

    // Build arrays
    let mut left_children = vec![-1i32; n_nodes];
    let mut right_children = vec![-1i32; n_nodes];
    let mut parents = vec![2147483647i32; n_nodes]; // max i32 = missing parent
    let mut split_indices = vec![0u32; n_nodes];
    let mut split_conditions = vec![0.0f64; n_nodes];
    let mut default_left = vec![false; n_nodes];
    let mut base_weights = vec![0.0f64; n_nodes];

    // Root parent stays as max i32
    for (i, n) in nodes.iter().enumerate() {
        match n {
            SerNode::Split { feat, thresh, left_idx, right_idx } => {
                split_indices[i] = *feat as u32;
                split_conditions[i] = *thresh;
                left_children[i] = *left_idx as i32;
                right_children[i] = *right_idx as i32;
                if *left_idx > 0 { parents[*left_idx] = i as i32; }
                if *right_idx > 0 { parents[*right_idx] = i as i32; }
            }
            SerNode::Leaf { weight } => {
                base_weights[i] = *weight;
            }
        }
    }

    format!(
        r#"{{"base_weights":{bw},"categories":[],"categories_nodes":[],"categories_segments":[],"categories_sizes":[],"default_left":{dl},"id":{tid},"left_children":{lc},"loss_changes":{ls},"parents":{ps},"right_children":{rc},"split_conditions":{sc},"split_indices":{si},"split_type":{st},"sum_hessian":{sh},"tree_param":{{"num_feature":"{nf}","num_nodes":"{nn}"}}}}"#,
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
        nf = split_indices.len().max(1).to_string(),
        nn = n_nodes.to_string(),
    )
}

enum SerNode {
    Split { feat: usize, thresh: f64, left_idx: usize, right_idx: usize },
    Leaf { weight: f64 },
}

fn serialize_node(
    tree: &TreeNode,
    nodes: &mut Vec<SerNode>,
    _stats: &mut Vec<f64>,
    _depth: usize,
) -> usize {
    let idx = nodes.len();
    match tree {
        TreeNode::Leaf { value } => {
            nodes.push(SerNode::Leaf { weight: *value });
        }
        TreeNode::Split { feature_index, threshold, left, right } => {
            nodes.push(SerNode::Split {
                feat: *feature_index,
                thresh: *threshold,
                left_idx: 0, // placeholder, will update
                right_idx: 0,
            });
            let left_idx = serialize_node(left, nodes, _stats, _depth + 1);
            let right_idx = serialize_node(right, nodes, _stats, _depth + 1);
            if let SerNode::Split { ref mut left_idx, ref mut right_idx, .. } = nodes[idx] {
                *left_idx = *left_idx;
                *right_idx = *right_idx;
            }
        }
    }
    idx
}

fn format_f64_array(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| {
        let s = format!("{:.6}", x).trim_end_matches('0').trim_end_matches('.').to_string();
        if s.is_empty() || s == "-" { "0.0".to_string() } else { s }
    }).collect();
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
    let items: Vec<String> = v.iter().map(|x| if *x { "true".to_string() } else { "false".to_string() }).collect();
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

        let ensemble = train_gbdt(&x, &y, &params);
        let r2 = ensemble.r_squared(&x, &y);
        assert!(r2 > 0.7, "GBDT R² too low: {r2}");

        let pred = ensemble.predict(&[3.0, 3.0]);
        assert!(pred.is_finite(), "pred not finite: {pred}");

        // Test serialization produces valid XGBoost JSON
        let json = ensemble.to_xgb_json();
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
}
