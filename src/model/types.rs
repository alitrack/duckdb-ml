use serde::{Deserialize, Serialize};

/// Supported ML algorithms
#[derive(
    Debug, Clone, Copy, PartialEq, Serialize, Deserialize, bincode::Encode, bincode::Decode,
)]
pub enum Algorithm {
    LinearRegression,
    RidgeRegression,
    LogisticRegression,
    Onnx,
    DecisionTreeRegressor,
    RandomForestRegressor,
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::LinearRegression => write!(f, "linear_regression"),
            Algorithm::RidgeRegression => write!(f, "ridge_regression"),
            Algorithm::LogisticRegression => write!(f, "logistic_regression"),
            Algorithm::Onnx => write!(f, "onnx"),
            Algorithm::DecisionTreeRegressor => write!(f, "decision_tree"),
            Algorithm::RandomForestRegressor => write!(f, "random_forest"),
        }
    }
}

impl Algorithm {
    pub fn parse_algorithm(s: &str) -> Option<Self> {
        match s {
            "linear_regression" => Some(Algorithm::LinearRegression),
            "ridge_regression" => Some(Algorithm::RidgeRegression),
            "logistic_regression" => Some(Algorithm::LogisticRegression),
            "onnx" => Some(Algorithm::Onnx),
            "decision_tree" => Some(Algorithm::DecisionTreeRegressor),
            "random_forest" => Some(Algorithm::RandomForestRegressor),
            _ => None,
        }
    }
}

/// Model metadata stored in duckdb_ml.models table
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct ModelMetadata {
    pub algorithm: Algorithm,
    pub num_features: usize,
    pub num_samples: usize,
    pub r_squared: Option<f64>,
    pub mse: Option<f64>,
    pub coefficients_count: usize,
    pub hyperparameters_json: String,
}

/// Core trait for all ML models
pub trait MlModel: Send + Sync {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError>;
    fn algorithm(&self) -> Algorithm;
    fn metadata(&self) -> &ModelMetadata;
    fn serialize(&self) -> Result<Vec<u8>, ModelError>;
    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized;
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Feature count mismatch: expected {expected}, got {got}")]
    FeatureCountMismatch { expected: usize, got: usize },
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Model not found: {0}")]
    NotFound(String),
    #[error("Training error: {0}")]
    Training(String),
}
