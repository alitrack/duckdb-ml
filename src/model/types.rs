use serde::{Deserialize, Serialize};

/// Supported ML algorithms
#[derive(
    Debug, Clone, Copy, PartialEq, Serialize, Deserialize, bincode::Encode, bincode::Decode,
)]
pub enum Algorithm {
    LinearRegression,
    RidgeRegression,
    ElasticNetRegression,
    LogisticRegression,
    MultinomialLogisticRegression,
    OrdinalLogisticRegression,
    RobustRegression,
    CoxProportionalHazards,
    Arima,
    Onnx,
    DecisionTreeRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
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
    DBSCAN,
    SVM,
    SVR,
    LDA,
    PolynomialRegression,
    FuzzyCMeans,
    AdaBoost,
    KaplanMeier,
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Algorithm::LinearRegression => write!(f, "linear_regression"),
            Algorithm::RidgeRegression => write!(f, "ridge_regression"),
            Algorithm::ElasticNetRegression => write!(f, "elastic_net"),
            Algorithm::LogisticRegression => write!(f, "logistic_regression"),
            Algorithm::MultinomialLogisticRegression => write!(f, "multilogistic"),
            Algorithm::OrdinalLogisticRegression => write!(f, "ordinal"),
            Algorithm::RobustRegression => write!(f, "robust"),
            Algorithm::CoxProportionalHazards => write!(f, "cox"),
            Algorithm::Arima => write!(f, "arima"),
            Algorithm::Onnx => write!(f, "onnx"),
            Algorithm::DecisionTreeRegressor => write!(f, "decision_tree"),
            Algorithm::RandomForestRegressor => write!(f, "random_forest"),
            Algorithm::RandomForestClassifier => write!(f, "rf_classifier"),
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
            Algorithm::DBSCAN => write!(f, "dbscan"),
            Algorithm::SVM => write!(f, "svm"),
            Algorithm::SVR => write!(f, "svr"),
            Algorithm::LDA => write!(f, "lda"),
            Algorithm::PolynomialRegression => write!(f, "polynomial_regression"),
            Algorithm::FuzzyCMeans => write!(f, "fuzzy_cmeans"),
            Algorithm::AdaBoost => write!(f, "adaboost"),
            Algorithm::KaplanMeier => write!(f, "kaplan_meier"),
        }
    }
}

impl Algorithm {
    pub fn parse_algorithm(s: &str) -> Option<Self> {
        match s {
            "linear_regression" => Some(Algorithm::LinearRegression),
            "ridge_regression" => Some(Algorithm::RidgeRegression),
            "elastic_net" => Some(Algorithm::ElasticNetRegression),
            "logistic_regression" => Some(Algorithm::LogisticRegression),
            "multilogistic" => Some(Algorithm::MultinomialLogisticRegression),
            "ordinal" => Some(Algorithm::OrdinalLogisticRegression),
            "robust" => Some(Algorithm::RobustRegression),
            "cox" => Some(Algorithm::CoxProportionalHazards),
            "arima" => Some(Algorithm::Arima),
            "onnx" => Some(Algorithm::Onnx),
            "decision_tree" => Some(Algorithm::DecisionTreeRegressor),
            "random_forest" => Some(Algorithm::RandomForestRegressor),
            "rf_classifier" => Some(Algorithm::RandomForestClassifier),
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
            "dbscan" => Some(Algorithm::DBSCAN),
            "svm" => Some(Algorithm::SVM),
            "svr" => Some(Algorithm::SVR),
            "lda" => Some(Algorithm::LDA),
            "polynomial_regression" => Some(Algorithm::PolynomialRegression),
            "fuzzy_cmeans" => Some(Algorithm::FuzzyCMeans),
            "adaboost" => Some(Algorithm::AdaBoost),
            "kaplan_meier" => Some(Algorithm::KaplanMeier),
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

    /// Embedding inference (AD-001/AD-002): returns the full f32 vector.
    ///
    /// Only encoder-type models (ONNX) implement this; the default errors so
    /// `ml_embed` over a regression model fails with a descriptive message.
    fn embed(&self, _features: &[f64]) -> Result<Vec<f32>, ModelError> {
        Err(ModelError::Training(format!(
            "model type {} does not support embedding (ONNX encoder required)",
            self.algorithm()
        )))
    }
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
