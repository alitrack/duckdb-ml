pub mod linear;
pub mod logistic;
pub mod registry;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use registry::ModelRegistry;
pub use registry::global_registry;

mod types;
pub use types::*;
