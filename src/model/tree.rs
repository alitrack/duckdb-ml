use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use crate::train::tree::{self, RandomForest, TreeNode};

/// Single Decision Tree regression model
pub struct TreeModel {
    pub metadata: ModelMetadata,
    tree: TreeNode,
}

impl TreeModel {
    pub fn new(tree: TreeNode, num_features: usize, num_samples: usize, max_depth: usize) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::DecisionTreeRegressor,
            num_features,
            num_samples,
            r_squared: None,
            mse: None,
            coefficients_count: 0,
            hyperparameters_json: serde_json::json!({"max_depth": max_depth}).to_string(),
        };
        Self { metadata, tree }
    }
}

impl MlModel for TreeModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        Ok(tree::predict_tree(&self.tree, features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::DecisionTreeRegressor
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        Ok(self.tree.to_bytes())
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let (tree, _) = TreeNode::from_bytes(blob)
            .ok_or_else(|| ModelError::Serialization("Failed to decode decision tree".into()))?;
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::DecisionTreeRegressor,
                num_features: 0, // reconstructed on load
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: 0,
                hyperparameters_json: "{}".into(),
            },
            tree,
        })
    }
}

/// Random Forest regression ensemble model
pub struct ForestModel {
    pub metadata: ModelMetadata,
    forest: RandomForest,
}

impl ForestModel {
    pub fn new(
        forest: RandomForest,
        num_features: usize,
        num_samples: usize,
        n_estimators: usize,
        max_depth: usize,
    ) -> Self {
        let metadata = ModelMetadata {
            algorithm: Algorithm::RandomForestRegressor,
            num_features,
            num_samples,
            r_squared: None,
            mse: None,
            coefficients_count: n_estimators,
            hyperparameters_json: serde_json::json!({
                "n_estimators": n_estimators,
                "max_depth": max_depth
            })
            .to_string(),
        };
        Self { metadata, forest }
    }
}

impl MlModel for ForestModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        if features.len() != self.metadata.num_features {
            return Err(ModelError::FeatureCountMismatch {
                expected: self.metadata.num_features,
                got: features.len(),
            });
        }
        Ok(self.forest.predict(features))
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::RandomForestRegressor
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        let mut buf = Vec::new();
        // Header: count of trees (4 bytes)
        let count = self.forest.trees.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());
        // Each tree serialized with its own encoding
        for tree in &self.forest.trees {
            let tree_bytes = tree.to_bytes();
            let len = tree_bytes.len() as u32;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&tree_bytes);
        }
        Ok(buf)
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        if blob.len() < 4 {
            return Err(ModelError::Serialization("Forest blob too short".into()));
        }
        let count = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;
        let mut trees = Vec::with_capacity(count);

        for _ in 0..count {
            if pos + 4 > blob.len() {
                return Err(ModelError::Serialization("Forest blob truncated".into()));
            }
            let len = u32::from_le_bytes(blob[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + len > blob.len() {
                return Err(ModelError::Serialization(
                    "Forest tree data truncated".into(),
                ));
            }
            let (tree, _) = TreeNode::from_bytes(&blob[pos..pos + len])
                .ok_or_else(|| ModelError::Serialization("Failed to decode forest tree".into()))?;
            trees.push(tree);
            pos += len;
        }

        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::RandomForestRegressor,
                num_features: 0,
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: trees.len(),
                hyperparameters_json: "{}".into(),
            },
            forest: RandomForest { trees },
        })
    }
}
