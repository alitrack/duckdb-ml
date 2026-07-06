use super::{Algorithm, MlModel, ModelError, ModelMetadata};
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub struct OnnxModel {
    pub metadata: ModelMetadata,
    file_path: String,
    session: Mutex<Session>,
}

impl std::fmt::Debug for OnnxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxModel")
            .field("metadata", &self.metadata)
            .field("file_path", &self.file_path)
            .finish()
    }
}

impl OnnxModel {
    pub fn new(file_path: &str, num_features: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(file_path)?;
        Ok(Self {
            metadata: ModelMetadata {
                algorithm: Algorithm::Onnx,
                num_features,
                num_samples: 0,
                r_squared: None,
                mse: None,
                coefficients_count: 0,
                hyperparameters_json: serde_json::json!({ "file_path": file_path }).to_string(),
            },
            file_path: file_path.into(),
            session: Mutex::new(session),
        })
    }
}

impl MlModel for OnnxModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let shape = vec![1usize, features.len()];
        let input = ort::value::Value::from_array((shape, features_f32))
            .map_err(|e| ModelError::Training(format!("ONNX input: {e}")))?;

        // Get output names (need to collect while session is borrowed)
        let first_name: String = {
            let session = self.session.lock().unwrap();
            session
                .outputs()
                .iter()
                .next()
                .map(|o| o.name().to_string())
                .unwrap_or_default()
        };

        // Run inference (lock again)
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| ModelError::Training(format!("ONNX run: {e}")))?;

        let output = outputs
            .get(&first_name)
            .ok_or_else(|| ModelError::Training("ONNX: output not found".into()))?;

        let (_shape, data): (&ort::value::Shape, &[f32]) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| ModelError::Training(format!("ONNX extract: {e}")))?;

        let result = data.first().copied().unwrap_or(0.0) as f64;
        Ok(result)
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::Onnx
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn serialize(&self) -> Result<Vec<u8>, ModelError> {
        let data = OnnxModelData {
            metadata: self.metadata.clone(),
            file_path: self.file_path.clone(),
        };
        bincode::encode_to_vec(&data, bincode::config::standard())
            .map_err(|e| ModelError::Serialization(e.to_string()))
    }

    fn deserialize(blob: &[u8]) -> Result<Self, ModelError>
    where
        Self: Sized,
    {
        let (data, _): (OnnxModelData, _) =
            bincode::decode_from_slice(blob, bincode::config::standard())
                .map_err(|e| ModelError::Serialization(e.to_string()))?;
        let session = Session::builder()
            .map_err(|e| ModelError::Serialization(format!("ONNX builder: {e}")))?
            .commit_from_file(&data.file_path)
            .map_err(|e| ModelError::Serialization(format!("ONNX load: {e}")))?;
        Ok(Self {
            metadata: data.metadata,
            file_path: data.file_path,
            session: Mutex::new(session),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
struct OnnxModelData {
    metadata: ModelMetadata,
    file_path: String,
}
