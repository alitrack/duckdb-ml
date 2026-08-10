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

    /// 嵌入模式推理（AD-002）：返回全量 f32 向量。
    ///
    /// - 输出按优先级选择：`pooler_output` → `text_embeds` → `last_hidden_state` → 首输出
    /// - `attention_mask` 输入按 session.inputs() 探测，存在则自动构造全 1 mask
    /// - `last_hidden_state`（3 维）取 first-token 的 hidden 切片
    pub fn embed(&self, features: &[f64]) -> Result<Vec<f32>, ModelError> {
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let shape = vec![1usize, features.len()];
        let main_input = ort::value::Value::from_array((shape.clone(), features_f32))
            .map_err(|e| ModelError::Training(format!("ONNX input: {e}")))?;

        let mut session = self.session.lock().unwrap();

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let first_input = input_names
            .first()
            .cloned()
            .ok_or_else(|| ModelError::Training("ONNX: model has no inputs".into()))?;
        let session_output_names = output_names(&session);
        let output_name = pick_output_name(&session_output_names)
            .ok_or_else(|| ModelError::Training("ONNX: no outputs".into()))?;

        let mut inputs = ort::inputs![first_input => main_input];
        if needs_attention_mask(&input_names) {
            let mask = ort::value::Value::from_array((shape, build_attention_mask(features.len())))
                .map_err(|e| ModelError::Training(format!("ONNX mask: {e}")))?;
            inputs.push(("attention_mask".into(), mask.into()));
        }

        let outputs = session
            .run(inputs)
            .map_err(|e| ModelError::Training(format!("ONNX run: {e}")))?;

        let (shape, data) = extract_output_tensor(&outputs, output_name)?;
        // shape: &ort::value::Shape（Deref 到 [i64]），自动解引用强转为 &[i64]
        Ok(extract_embedding(shape, data))
    }
}

// ── AD-002: 输出兼容层（与模型类型解耦，供任意 ONNX 模型复用）──

/// 输出提取模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxOutputMode {
    /// 标量预测：取输出张量第一个值
    Predict,
    /// 嵌入向量：取全量 f32 切片
    Embed,
}

/// 输出选择优先级：`pooler_output` → `text_embeds` → `last_hidden_state` → 首输出
const OUTPUT_PRIORITY: [&str; 3] = ["pooler_output", "text_embeds", "last_hidden_state"];

/// 按优先级选择输出名；无匹配回退首输出；空列表返回 None（契约 1 / 错误路径）
fn pick_output_name(names: &[String]) -> Option<&str> {
    for priority in OUTPUT_PRIORITY {
        if names.iter().any(|n| n == priority) {
            return Some(priority);
        }
    }
    names.first().map(String::as_str)
}

/// 从 run 结果提取指定输出的 f32 张量（共享提取路径）
///
/// `SessionOutputs` 自带内部生命周期，签名含 2 个生命周期 → 省略规则不适用，
/// 显式 `'a` 必需（clippy::needless_lifetimes 在此为误报）。
#[allow(clippy::needless_lifetimes)]
fn extract_output_tensor<'a>(
    outputs: &'a ort::session::SessionOutputs,
    output_name: impl AsRef<str>,
) -> Result<(&'a ort::value::Shape, &'a [f32]), ModelError> {
    let output_name = output_name.as_ref();
    let output = outputs
        .get(output_name)
        .ok_or_else(|| ModelError::Training(format!("ONNX: output {output_name} not found")))?;
    output
        .try_extract_tensor::<f32>()
        .map_err(|e| ModelError::Training(format!("ONNX extract: {e}")))
}

/// 提取嵌入向量：1D/2D 取全量；3D（1, seq, hidden）取 first-token hidden 切片（契约 3）
fn extract_embedding(shape: &[i64], data: &[f32]) -> Vec<f32> {
    if shape.len() >= 3 {
        if let Some(&hidden) = shape.last().filter(|&&d| d > 0) {
            let end = (hidden as usize).min(data.len());
            return data[..end].to_vec();
        }
    }
    data.to_vec()
}

/// 构造全 1 attention mask（契约 2 边界：长度 = 主输入长度）
fn build_attention_mask(len: usize) -> Vec<i64> {
    vec![1i64; len]
}

/// 探测 session 输入是否需要 attention_mask（契约 2）
fn needs_attention_mask(input_names: &[String]) -> bool {
    input_names.iter().any(|n| n == "attention_mask")
}

/// 收集 session 输出名（借用期间收集，避免锁内二次借用）
fn output_names(session: &Session) -> Vec<String> {
    session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect()
}

impl MlModel for OnnxModel {
    fn predict(&self, features: &[f64]) -> Result<f64, ModelError> {
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let shape = vec![1usize, features.len()];
        let input = ort::value::Value::from_array((shape, features_f32))
            .map_err(|e| ModelError::Training(format!("ONNX input: {e}")))?;

        // Run inference
        let mut session = self.session.lock().unwrap();
        let session_output_names = output_names(&session);
        let output_name = pick_output_name(&session_output_names)
            .ok_or_else(|| ModelError::Training("ONNX: no outputs".into()))?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| ModelError::Training(format!("ONNX run: {e}")))?;

        let (_shape, data) = extract_output_tensor(&outputs, output_name)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_output_prioritizes_pooler_output() {
        let names = vec!["logits".to_string(), "pooler_output".to_string()];
        assert_eq!(pick_output_name(&names), Some("pooler_output"));
    }

    #[test]
    fn pick_output_prioritizes_text_embeds_when_no_pooler() {
        let names = vec!["logits".to_string(), "text_embeds".to_string()];
        assert_eq!(pick_output_name(&names), Some("text_embeds"));
    }

    #[test]
    fn pick_output_falls_back_to_last_hidden_state() {
        let names = vec!["last_hidden_state".to_string(), "logits".to_string()];
        assert_eq!(pick_output_name(&names), Some("last_hidden_state"));
    }

    #[test]
    fn pick_output_falls_back_to_first_output() {
        let names = vec!["output_0".to_string(), "output_1".to_string()];
        assert_eq!(pick_output_name(&names), Some("output_0"));
    }

    #[test]
    fn pick_output_empty_returns_none() {
        let names: Vec<String> = vec![];
        assert_eq!(pick_output_name(&names), None);
    }

    #[test]
    fn extract_embedding_2d_returns_full_vector() {
        let shape = vec![1i64, 512];
        let data: Vec<f32> = (0..512).map(|i| i as f32).collect();
        assert_eq!(extract_embedding(&shape, &data), data);
    }

    #[test]
    fn extract_embedding_3d_takes_first_token_slice() {
        // (1, seq=7, hidden=512) → 只取 first token 的 512 维
        let shape = vec![1i64, 7, 512];
        let data: Vec<f32> = (0..7 * 512).map(|i| i as f32).collect();
        let emb = extract_embedding(&shape, &data);
        assert_eq!(emb.len(), 512);
        assert_eq!(emb, &data[..512]);
    }

    #[test]
    fn extract_embedding_3d_zero_hidden_returns_all() {
        let shape = vec![1i64, 7, 0];
        let data = vec![1.0f32, 2.0, 3.0];
        assert_eq!(extract_embedding(&shape, &data), data);
    }

    #[test]
    fn extract_embedding_guards_hidden_larger_than_data() {
        let shape = vec![1i64, 10, 4096];
        let data = vec![5.0f32; 100];
        let emb = extract_embedding(&shape, &data);
        assert_eq!(emb.len(), 100);
    }

    #[test]
    fn attention_mask_is_all_ones() {
        assert_eq!(build_attention_mask(4), vec![1i64; 4]);
        assert_eq!(build_attention_mask(0), Vec::<i64>::new());
    }

    #[test]
    fn needs_attention_mask_detects_input() {
        let names = vec!["input_ids".to_string(), "attention_mask".to_string()];
        assert!(needs_attention_mask(&names));
        let without = vec!["pixel_values".to_string()];
        assert!(!needs_attention_mask(&without));
        assert!(!needs_attention_mask(&[]));
    }
}
