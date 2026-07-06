pub mod kmeans;
pub mod knn;
pub mod linear;
pub mod logistic;
pub mod naive_bayes;
pub mod pca;
pub mod registry;
pub mod tree;
pub mod xgboost;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use registry::ModelRegistry;
pub use registry::global_registry;

mod types;
pub use types::*;
