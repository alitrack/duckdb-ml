pub mod adaboost;
pub mod arima;
pub mod cox;
pub mod dbscan;
pub mod fcm;
pub mod kmeans;
pub mod km;
pub mod knn;
pub mod lasso;
pub mod lda;
pub mod linear;
pub mod logistic;
pub mod mlp;
pub mod multilogistic;
pub mod naive_bayes;
pub mod ordinal;
pub mod pca;
pub mod poly;
pub mod registry;
pub mod svm;
pub mod svr;
pub mod tree;
pub mod xgboost;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use registry::global_registry;
pub use registry::ModelRegistry;

mod types;
pub use types::*;
