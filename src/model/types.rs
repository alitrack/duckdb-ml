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
    KMeans,
    XGBoostRegressor,
    XGBoostClassifier,
    XGBoostRegression,
    XGBoostBinary,
    KNNRegressor,
    KNNClassifier,
    NaiveBayes,
    PCA,
    LassoRegression,
    MlpRegressor,
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
            Algorithm::KMeans => write!(f, "kmeans"),
            Algorithm::XGBoostRegressor => write!(f, "xgboost_regressor"),
            Algorithm::XGBoostClassifier => write!(f, "xgboost_classifier"),
            Algorithm::XGBoostRegression => write!(f, "xgboost_regression"),
            Algorithm::XGBoostBinary => write!(f, "xgboost_binary"),
            Algorithm::KNNRegressor => write!(f, "knn_regressor"),
            Algorithm::KNNClassifier => write!(f, "knn_classifier"),
            Algorithm::NaiveBayes => write!(f, "naive_bayes"),
            Algorithm::PCA => write!(f, "pca"),
            Algorithm::LassoRegression => write!(f, "lasso_regression"),
            Algorithm::MlpRegressor => write!(f, "mlp_regressor"),
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
            "kmeans" => Some(Algorithm::KMeans),
            "xgboost_regressor" => Some(Algorithm::XGBoostRegressor),
            "xgboost_classifier" => Some(Algorithm::XGBoostClassifier),
            "xgboost_regression" => Some(Algorithm::XGBoostRegression),
            "xgboost_binary" => Some(Algorithm::XGBoostBinary),
            "knn_regressor" => Some(Algorithm::KNNRegressor),
            "knn_classifier" => Some(Algorithm::KNNClassifier),
            "naive_bayes" => Some(Algorithm::NaiveBayes),
            "pca" => Some(Algorithm::PCA),
            "lasso_regression" => Some(Algorithm::LassoRegression),
            "mlp_regressor" => Some(Algorithm::MlpRegressor),
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
