pub mod linear;
pub mod logistic;
pub mod registry;

pub use registry::ModelRegistry;
pub use registry::global_registry;

mod types;
pub use types::*;
