//! CART Decision Tree and Random Forest training engine
//!
//! Pure Rust, zero external dependencies beyond std.

/// A single node in a decision tree
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Leaf node — prediction value
    Leaf { value: f64 },
    /// Internal split node
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Decision tree hyperparameters
#[derive(Debug, Clone)]
pub struct TreeParams {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>, // None = use all features
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            max_depth: 10,
            min_samples_split: 5,
            min_samples_leaf: 2,
            max_features: None,
        }
    }
}

/// Build a single CART regression tree
pub fn build_tree(x: &[Vec<f64>], y: &[f64], params: &TreeParams) -> TreeNode {
    let indices: Vec<usize> = (0..x.len()).collect();
    build_node(x, y, &indices, params, 0)
}

fn build_node(
    x: &[Vec<f64>],
    y: &[f64],
    indices: &[usize],
    params: &TreeParams,
    depth: usize,
) -> TreeNode {
    // Compute leaf value: mean of targets
    let leaf_value = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len() as f64;

    // Stop criteria
    if indices.len() < params.min_samples_split
        || depth >= params.max_depth
        || indices.len() < 2 * params.min_samples_leaf
    {
        return TreeNode::Leaf { value: leaf_value };
    }

    // Check if all targets are identical
    let first_y = y[indices[0]];
    if indices.iter().all(|&i| (y[i] - first_y).abs() < 1e-12) {
        return TreeNode::Leaf { value: leaf_value };
    }

    let n_features = x[0].len();

    // Determine which features to consider at this split
    let feature_subset: Vec<usize> = match params.max_features {
        Some(k) => {
            let n = k.min(n_features);
            random_subset(n_features, n)
        }
        None => (0..n_features).collect(),
    };

    // Find best split
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_mse = f64::MAX;
    let mut best_left: Vec<usize> = vec![];
    let mut best_right: Vec<usize> = vec![];

    for &f_idx in &feature_subset {
        // Collect unique sorted values for this feature
        let mut values: Vec<f64> = indices.iter().map(|&i| x[i][f_idx]).collect();
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();

        // Try each mid-point as threshold
        for w in values.windows(2) {
            let threshold = (w[0] + w[1]) / 2.0;

            let mut left: Vec<usize> = Vec::new();
            let mut right: Vec<usize> = Vec::new();
            for &i in indices {
                if x[i][f_idx] <= threshold {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }

            // Skip degenerate splits
            if left.len() < params.min_samples_leaf || right.len() < params.min_samples_leaf {
                continue;
            }

            // Compute MSE for this split
            let left_mean = left.iter().map(|&i| y[i]).sum::<f64>() / left.len() as f64;
            let right_mean = right.iter().map(|&i| y[i]).sum::<f64>() / right.len() as f64;
            let mse = left
                .iter()
                .map(|&i| (y[i] - left_mean).powi(2))
                .sum::<f64>()
                + right
                    .iter()
                    .map(|&i| (y[i] - right_mean).powi(2))
                    .sum::<f64>();

            if mse < best_mse {
                best_mse = mse;
                best_feature = f_idx;
                best_threshold = threshold;
                best_left = left;
                best_right = right;
            }
        }
    }

    // If no valid split found, return leaf
    if best_left.is_empty() || best_right.is_empty() {
        return TreeNode::Leaf { value: leaf_value };
    }

    // Recurse
    let left_child = build_node(x, y, &best_left, params, depth + 1);
    let right_child = build_node(x, y, &best_right, params, depth + 1);

    TreeNode::Split {
        feature_index: best_feature,
        threshold: best_threshold,
        left: Box::new(left_child),
        right: Box::new(right_child),
    }
}

/// Predict from a single decision tree
pub fn predict_tree(node: &TreeNode, features: &[f64]) -> f64 {
    match node {
        TreeNode::Leaf { value } => *value,
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            if features[*feature_index] <= *threshold {
                predict_tree(left, features)
            } else {
                predict_tree(right, features)
            }
        }
    }
}

/// Random Forest ensemble
pub struct RandomForest {
    pub trees: Vec<TreeNode>,
}

impl RandomForest {
    /// Train a random forest
    pub fn train(x: &[Vec<f64>], y: &[f64], n_estimators: usize, params: &TreeParams) -> Self {
        let n_features = x[0].len();
        let n_samples = x.len();

        // max_features for regression: n_features / 3 (rounded up)
        let max_features = n_features.div_ceil(3);

        let mut tree_params = params.clone();
        tree_params.max_features = Some(max_features);

        let mut trees = Vec::with_capacity(n_estimators);

        for _ in 0..n_estimators {
            // Bootstrap sample (with replacement)
            let (bs_x, bs_y) = bootstrap_sample(x, y, n_samples);

            let tree = build_tree(&bs_x, &bs_y, &tree_params);
            trees.push(tree);
        }

        Self { trees }
    }

    /// Predict: average of all tree predictions
    pub fn predict(&self, features: &[f64]) -> f64 {
        let sum: f64 = self.trees.iter().map(|t| predict_tree(t, features)).sum();
        sum / self.trees.len() as f64
    }
}

/// Generate a bootstrap sample (with replacement)
fn bootstrap_sample(x: &[Vec<f64>], y: &[f64], n_samples: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut out_x = Vec::with_capacity(n_samples);
    let mut out_y = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let idx = fast_rand() % n_samples;
        out_x.push(x[idx].clone());
        out_y.push(y[idx]);
    }

    (out_x, out_y)
}

/// Simple PRNG — Xorshift
static XORSHIFT_STATE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0xABCD_EF01_2345_6789);

#[inline]
fn fast_rand() -> usize {
    use std::sync::atomic::Ordering;
    let mut x = XORSHIFT_STATE.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    XORSHIFT_STATE.store(x, Ordering::Relaxed);
    x as usize
}

/// Select k random indices from 0..n without replacement (Fisher-Yates partial)
fn random_subset(n: usize, k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k.min(n) {
        let j = i + (fast_rand() % (n - i));
        indices.swap(i, j);
    }
    indices[..k.min(n)].to_vec()
}

// --- Serialization helpers for TreeNode ---

impl TreeNode {
    /// Serialize tree to bytes using a simple pre-order encoding
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.encode(&mut buf);
        buf
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            TreeNode::Leaf { value } => {
                buf.push(0); // tag: leaf
                buf.extend_from_slice(&value.to_le_bytes());
            }
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                buf.push(1); // tag: split
                buf.extend_from_slice(&(*feature_index as u32).to_le_bytes());
                buf.extend_from_slice(&threshold.to_le_bytes());
                left.encode(buf);
                right.encode(buf);
            }
        }
    }

    /// Deserialize tree from bytes (pre-order encoding)
    pub fn from_bytes(data: &[u8]) -> Option<(TreeNode, usize)> {
        let mut pos = 0;
        let tree = Self::decode(data, &mut pos)?;
        Some((tree, pos))
    }

    fn decode(data: &[u8], pos: &mut usize) -> Option<TreeNode> {
        if *pos >= data.len() {
            return None;
        }
        match data[*pos] {
            0 => {
                // leaf
                *pos += 1;
                if *pos + 8 > data.len() {
                    return None;
                }
                let bytes: [u8; 8] = data[*pos..*pos + 8].try_into().ok()?;
                *pos += 8;
                let value = f64::from_le_bytes(bytes);
                Some(TreeNode::Leaf { value })
            }
            1 => {
                // split
                *pos += 1;
                if *pos + 12 > data.len() {
                    return None;
                }
                let fi_bytes: [u8; 4] = data[*pos..*pos + 4].try_into().ok()?;
                *pos += 4;
                let feature_index = u32::from_le_bytes(fi_bytes) as usize;

                let th_bytes: [u8; 8] = data[*pos..*pos + 8].try_into().ok()?;
                *pos += 8;
                let threshold = f64::from_le_bytes(th_bytes);

                let left = Box::new(Self::decode(data, pos)?);
                let right = Box::new(Self::decode(data, pos)?);

                Some(TreeNode::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_serialization_roundtrip() {
        let tree = TreeNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf { value: 1.0 }),
            right: Box::new(TreeNode::Leaf { value: 2.0 }),
        };
        let bytes = tree.to_bytes();
        let (recovered, _) = TreeNode::from_bytes(&bytes).unwrap();

        match recovered {
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                assert_eq!(feature_index, 0);
                assert!((threshold - 0.5).abs() < 1e-10);
                match *left {
                    TreeNode::Leaf { value } => assert!((value - 1.0).abs() < 1e-10),
                    _ => panic!("Expected leaf"),
                }
                match *right {
                    TreeNode::Leaf { value } => assert!((value - 2.0).abs() < 1e-10),
                    _ => panic!("Expected leaf"),
                }
            }
            _ => panic!("Expected split node"),
        }
    }

    #[test]
    fn test_simple_regression_tree() {
        // y = 2*x1 + 3*x2, linear separable with enough depth
        let x: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![i as f64 % 10.0, (i as f64 / 10.0).floor()])
            .collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0 * r[1]).collect();

        // Deep tree to fit tightly to training data
        let params = TreeParams {
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
        };
        let tree = build_tree(&x, &y, &params);

        // Test a few predictions — tree is piecewise constant so allow small error
        let pred = predict_tree(&tree, &[5.0, 5.0]);
        let expected = 2.0 * 5.0 + 3.0 * 5.0;
        assert!(
            (pred - expected).abs() < 3.0,
            "pred={pred}, expected={expected}"
        );

        // Average error across dataset should be small
        let mse: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(row, &true_y)| {
                let p = predict_tree(&tree, row);
                (p - true_y).powi(2)
            })
            .sum::<f64>()
            / x.len() as f64;
        assert!(mse < 5.0, "MSE too high: {mse}");
    }

    #[test]
    fn test_random_forest_regression() {
        // Simple clean dataset: y = x0 + x1
        let x: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
        ];
        // Repeat to give enough samples
        let x: Vec<Vec<f64>> = x.iter().cycle().take(x.len() * 50).cloned().collect();
        let y: Vec<f64> = x.iter().map(|r| r[0] + r[1]).collect();

        let params = TreeParams {
            max_depth: 5,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None, // forest sets this internally
        };
        let rf = RandomForest::train(&x, &y, 30, &params);

        let pred = rf.predict(&[5.0, 5.0]);
        assert!((pred - 10.0).abs() < 3.0, "pred={pred}, expected ~10");

        let pred2 = rf.predict(&[0.0, 0.0]);
        assert!((pred2 - 0.0).abs() < 2.0, "pred={pred2}, expected ~0");
    }
}
