//! Embedding storage & similarity retrieval (AD-001).
//!
//! SQL surface:
//!
//! ```sql
//! -- Embed (scalar, onnx feature): f32 LE packed blob, store in any column
//! UPDATE media SET embeds = ml_embed('clip', features_json);
//!
//! -- Similarity (scalar): full-scan cosine over candidates from a subquery
//! -- (table-function params can't contain subqueries; scalar params can —
//! -- same pattern as ml_predict_batch_value / ml_ols):
//! SELECT ml_similarity_value(
//!     '[0.1,0.2,...]',
//!     (SELECT to_json(list({'row_id': id, 'embeds': embeds})) FROM media),
//!     10, 0.3);
//! ```
//!
//! Candidate JSON element forms (auto-detected):
//! - `{"row_id": N, "embeds": "0102..."}`   struct, embeds as hex blob (f32 LE)
//! - `{"row_id": N, "embeds": [f32,...]}`   struct, embeds as vector
//! - `[N, "0102..."]` / `[N, [f32,...]]`    pair form
//! - `[N, null]`                            skipped (NULL embedding)
//!
//! Score = cosine similarity of L2-normalized vectors, clamped to [-1, 1].
//! Rows with embedding length % 4 != 0 are skipped (not panic); dimension
//! mismatches score 0.0. `k` capped at 10000; threshold filters before top-k.
//!
//! Cancellation: a global atomic flag (set via [`set_similarity_cancel`], for
//! embedded/rlib use) aborts the scan, returning partial top-k with
//! `"cancelled": true`. At the SQL layer, DuckDB's own query interrupt
//! terminates the statement — progress events are not emittable through the
//! scalar surface (documented deviation, see tasks.md).

use arrow::array::{Array, ArrayRef, BinaryArray, Float64Array, StringArray, UInt64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use duckdb::vscalar::arrow::{ArrowFunctionSignature, VArrowScalar};
use serde_json::Value;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;

// ── f32 LE blob packing (storage contract 1) ──

/// Pack f32 vector as little-endian bytes (4 × dim).
pub fn pack_f32_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Unpack f32 LE bytes; `None` when length is not a multiple of 4.
pub fn unpack_f32_le(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

/// Decode a hex string to bytes (`"0102"` → `[0x01, 0x02]`).
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for pair in s.as_bytes().chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

/// Decode DuckDB's JSON binary-string format: printable ASCII passes through,
/// `\xNN` escapes decode to bytes (`"\\x00\\x00\\x80?"` → `[0,0,128,63]`).
/// This is what `to_json(blob_col)` emits for BLOB columns.
fn decode_duckdb_binary(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                'x' => {
                    let hi = chars.next()?.to_digit(16)?;
                    let lo = chars.next()?.to_digit(16)?;
                    out.push(((hi << 4) | lo) as u8);
                }
                c2 => out.push(c2 as u8), // `\\` `\"` etc.
            }
        } else if c.is_ascii() {
            out.push(c as u8);
        } else {
            return None; // non-ASCII should always be escaped by DuckDB
        }
    }
    Some(out)
}

/// Decode a blob string: hex (`"0000803F"`), DuckDB JSON binary-string
/// (`"\\x00\\x00\\x80?"`), or raw printable bytes.
fn blob_from_string(s: &str) -> Option<Vec<u8>> {
    if s.contains('\\') {
        return decode_duckdb_binary(s);
    }
    if s.len().is_multiple_of(2) && s.chars().all(|c| c.is_ascii_hexdigit()) {
        return hex_decode(s);
    }
    decode_duckdb_binary(s)
}

/// L2-normalize; `None` for zero-norm or non-finite vectors.
fn l2_normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return None;
    }
    Some(v.iter().map(|x| x / norm).collect())
}

/// Cosine similarity between two vectors (both normalized inside).
/// Clamped to [-1, 1]; NaN/Inf input or dimension mismatch → 0.0.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (Some(an), Some(bn)) = (l2_normalize(a), l2_normalize(b)) else {
        return 0.0;
    };
    let dot: f32 = an.iter().zip(bn.iter()).map(|(x, y)| x * y).sum();
    if !dot.is_finite() {
        0.0
    } else {
        dot.clamp(-1.0, 1.0)
    }
}

// ── Query / candidate parsing ──

/// Parse the query embedding: JSON array `[f32,...]` or hex blob `"0102..."`.
fn parse_query_vec(query: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let q = query.trim();
    if q.is_empty() {
        return Err("query embedding is empty".into());
    }
    if q.starts_with('[') {
        let v: Vec<f64> = serde_json::from_str(q)
            .map_err(|e| format!("invalid query embedding JSON '{q}': {e}"))?;
        if v.is_empty() {
            return Err("query embedding is empty".into());
        }
        return Ok(v.iter().map(|&x| x as f32).collect());
    }
    // blob form: DuckDB JSON binary-string or hex
    let bytes = blob_from_string(q).ok_or("query embedding blob is malformed")?;
    let v = unpack_f32_le(&bytes).ok_or("query embedding blob length is not a multiple of 4")?;
    if v.is_empty() {
        return Err("query embedding is empty".into());
    }
    Ok(v)
}

/// Extract the embedding payload from one candidate element (JSON value).
/// Returns `(Option<Vec<f32>>, is_null)` — `Ok((None, _))` marks a skipped
/// (NULL) embedding, `Err` marks a malformed-but-countable row.
fn candidate_embed(value: &Value) -> Result<(Option<Vec<f32>>, bool), ()> {
    let embeds = match value {
        Value::Array(pair) if pair.len() == 2 => &pair[1],
        Value::Object(map) => map.get("embeds").ok_or(())?,
        _ => return Err(()),
    };
    match embeds {
        Value::Null => Ok((None, true)),
        Value::String(s) => {
            let bytes = blob_from_string(s).ok_or(())?;
            let v = unpack_f32_le(&bytes).ok_or(())?;
            Ok((Some(v), false))
        }
        Value::Array(vec) => {
            let v: Result<Vec<f32>, _> = vec
                .iter()
                .map(|x| x.as_f64().map(|f| f as f32).ok_or(()))
                .collect();
            Ok((Some(v.map_err(|_| ())?), false))
        }
        _ => Err(()),
    }
}

fn candidate_row_id(value: &Value) -> Option<Value> {
    match value {
        Value::Array(pair) if pair.len() == 2 => Some(pair[0].clone()),
        Value::Object(map) => map.get("row_id").cloned(),
        _ => None,
    }
}

/// Deterministic tie-break for row ids: numeric ids compare numerically.
fn row_id_cmp(a: &Value, b: &Value) -> Ordering {
    match (a.as_u64(), b.as_u64()) {
        (Some(x), Some(y)) => x.cmp(&y),
        _ => a.to_string().cmp(&b.to_string()),
    }
}

// ── Top-K scan core ──

const K_CAP: usize = 10_000;
const CANCEL_CHECK_INTERVAL: usize = 4096;

/// Global cancellation flag for in-flight similarity scans (embedded use).
static SIMILARITY_CANCEL: AtomicBool = AtomicBool::new(false);

/// Set/reset the global similarity-scan cancellation flag.
pub fn set_similarity_cancel(cancel: bool) {
    SIMILARITY_CANCEL.store(cancel, AtomicOrdering::Relaxed);
}

#[derive(PartialEq)]
struct HeapEntry {
    score: f32,
    row_id: Value,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| row_id_cmp(&self.row_id, &other.row_id))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Full-scan cosine similarity over JSON candidates.
///
/// Returns a JSON object:
/// ```json
/// {"results":[{"row_id":1,"score":0.99},...], "scanned":N,
///  "skipped_null":a, "skipped_bad_len":b, "skipped_dim":c, "cancelled":false}
/// ```
pub fn similarity_from_json(
    query: &str,
    candidates: &str,
    k: usize,
    threshold: f64,
) -> Result<String, Box<dyn Error>> {
    if k == 0 {
        return Err("k must be >= 1".into());
    }
    let k = k.min(K_CAP);
    let threshold = threshold.clamp(-1.0, 1.0) as f32;

    let q = parse_query_vec(query)?;
    let q_norm = l2_normalize(&q)
        .ok_or_else(|| "query embedding is not normalizable (zero norm or NaN)".to_string())?;

    let parsed: Value = if candidates.trim().is_empty() {
        Value::Array(vec![])
    } else {
        serde_json::from_str(candidates).map_err(|e| format!("invalid candidates JSON: {e}"))?
    };
    let arr = parsed.as_array().ok_or("candidates must be a JSON array")?;

    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
    let mut scanned = 0usize;
    let mut skipped_null = 0usize;
    let mut skipped_bad_len = 0usize;
    let mut skipped_dim = 0usize;
    let mut cancelled = false;

    for (i, el) in arr.iter().enumerate() {
        // Cancel check from the first interval boundary onward — a small scan
        // (< INTERVAL rows) is never interrupted, which keeps unit tests
        // deterministic even when a parallel test holds the flag set.
        if i > 0
            && i % CANCEL_CHECK_INTERVAL == 0
            && SIMILARITY_CANCEL.load(AtomicOrdering::Relaxed)
        {
            cancelled = true;
            break;
        }
        scanned += 1;
        let row_id = match candidate_row_id(el) {
            Some(id) => id,
            None => {
                skipped_bad_len += 1; // malformed element, counted, not fatal
                continue;
            }
        };
        match candidate_embed(el) {
            Ok((None, true)) => {
                skipped_null += 1;
                continue;
            }
            Ok((Some(v), _)) => {
                let score = if v.len() != q_norm.len() {
                    skipped_dim += 1;
                    0.0
                } else {
                    cosine_similarity(&q, &v) // L2-normalize both, dot, clamp
                };
                if score >= threshold {
                    heap.push(Reverse(HeapEntry { score, row_id }));
                    if heap.len() > k {
                        heap.pop(); // min-heap: drop the smallest; keep k largest
                    }
                }
            }
            Ok((None, false)) => {
                skipped_bad_len += 1; // empty vector
                continue;
            }
            Err(()) => {
                skipped_bad_len += 1; // malformed embedding payload
                continue;
            }
        }
    }

    // Sort: score desc, row_id asc (deterministic).
    let mut entries: Vec<HeapEntry> = heap.into_iter().map(|Reverse(e)| e).collect();
    entries.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| row_id_cmp(&a.row_id, &b.row_id))
    });

    let results: Vec<Value> = entries
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "row_id": e.row_id,
                "score": (e.score as f64 * 1e6).round() / 1e6,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "results": results,
        "scanned": scanned,
        "skipped_null": skipped_null,
        "skipped_bad_len": skipped_bad_len,
        "skipped_dim": skipped_dim,
        "cancelled": cancelled,
    })
    .to_string())
}

// ── Shared scalar helpers ──

fn col_str(input: &RecordBatch, idx: usize) -> Result<&StringArray, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("expected VARCHAR column".into())
}

fn col_u64(input: &RecordBatch, idx: usize) -> Result<&UInt64Array, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or("expected UBIGINT column".into())
}

fn col_f64(input: &RecordBatch, idx: usize) -> Result<&Float64Array, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or("expected DOUBLE column".into())
}

// ── ml_embed(model, features_json) → BLOB ──

/// Embedding inference per row: model + JSON feature vector → f32 LE blob.
pub struct EmbedFn;

impl VArrowScalar for EmbedFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let model_col = col_str(&input, 0)?;
        let features_col = col_str(&input, 1)?;

        // Model name is constant across rows — resolve once.
        let model_name = if model_col.is_null(0) {
            return Err("NULL model name".into());
        } else {
            model_col.value(0)
        };
        let arc_model = crate::model::global_registry()
            .get_deployed_model(model_name)
            .or_else(|| crate::model::global_registry().get(model_name))
            .ok_or_else(|| format!("Model '{model_name}' not loaded"))?;

        let mut blobs: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
        for i in 0..n {
            if features_col.is_null(i) {
                blobs.push(None);
                continue;
            }
            let features_json = features_col.value(i);
            let x: Vec<f64> = serde_json::from_str(features_json)
                .map_err(|e| format!("Invalid features JSON '{features_json}': {e}"))?;
            let emb = arc_model.embed(&x)?;
            blobs.push(Some(pack_f32_le(&emb)));
        }

        let out: Vec<Option<&[u8]>> = blobs.iter().map(|b| b.as_deref()).collect();
        let arr = BinaryArray::from_opt_vec(out);
        Ok(Arc::new(arr))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![DataType::Utf8, DataType::Utf8],
            DataType::Binary,
        )]
    }
}

// ── ml_similarity_value(query, candidates, [k], [threshold]) → VARCHAR ──

/// Full-scan cosine similarity; returns the top-k JSON payload (see
/// [`similarity_from_json`]).
pub struct SimilarityFn;

impl VArrowScalar for SimilarityFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let n_args = input.num_columns();
        if !(2..=4).contains(&n_args) {
            return Err(
                "ml_similarity_value requires 2..4 args: query, candidates[, k[, threshold]]"
                    .into(),
            );
        }

        let query_col = col_str(&input, 0)?;
        let candidates_col = col_str(&input, 1)?;

        let k: usize = if n_args >= 3 {
            let col = col_u64(&input, 2)?;
            if col.is_null(0) {
                10
            } else {
                col.value(0) as usize
            }
        } else {
            10
        };

        let threshold: f64 = if n_args >= 4 {
            let col = col_f64(&input, 3)?;
            if col.is_null(0) {
                0.0
            } else {
                col.value(0)
            }
        } else {
            0.0
        };

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if query_col.is_null(i) || candidates_col.is_null(i) {
                out.push(None);
                continue;
            }
            let json =
                similarity_from_json(query_col.value(i), candidates_col.value(i), k, threshold)?;
            out.push(Some(json));
        }
        let arr = StringArray::from(out);
        Ok(Arc::new(arr))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![
            ArrowFunctionSignature::exact(vec![DataType::Utf8, DataType::Utf8], DataType::Utf8),
            ArrowFunctionSignature::exact(
                vec![DataType::Utf8, DataType::Utf8, DataType::UInt64],
                DataType::Utf8,
            ),
            ArrowFunctionSignature::exact(
                vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::UInt64,
                    DataType::Float64,
                ],
                DataType::Utf8,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── blob packing ──

    #[test]
    fn pack_unpack_roundtrip() {
        let v = vec![1.5f32, -2.25, 0.0, f32::MAX, f32::MIN_POSITIVE];
        let bytes = pack_f32_le(&v);
        assert_eq!(bytes.len(), v.len() * 4);
        assert_eq!(unpack_f32_le(&bytes).unwrap(), v);
    }

    #[test]
    fn unpack_rejects_non_multiple_of_4() {
        assert_eq!(unpack_f32_le(&[0u8, 1, 2]), None);
        assert_eq!(unpack_f32_le(&[]), Some(vec![]));
    }

    #[test]
    fn pack_is_little_endian() {
        // 1.0f32 LE = 00 00 80 3F
        assert_eq!(pack_f32_le(&[1.0]), vec![0x00, 0x00, 0x80, 0x3F]);
    }

    // ── cosine ──

    #[test]
    fn cosine_identical_is_one() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_opposite_is_minus_one() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
    }

    #[test]
    fn cosine_scale_invariant() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        let s = cosine_similarity(&a, &b);
        assert!((s - 1.0).abs() < 1e-6, "s={s}");
    }

    #[test]
    fn cosine_dim_mismatch_is_zero() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_nan_is_zero() {
        assert_eq!(cosine_similarity(&[f32::NAN, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[f32::INFINITY, 0.0]), 0.0);
    }

    #[test]
    fn cosine_clamped() {
        // near-1 float error must clamp to exactly 1.0
        let a = [1.0f32, 1e-7];
        let b = [1.0f32, 1e-7];
        let s = cosine_similarity(&a, &b);
        assert!(s <= 1.0 && s >= -1.0);
    }

    #[test]
    fn normalize_zero_norm_is_none() {
        assert_eq!(l2_normalize(&[0.0, 0.0]), None);
        assert_eq!(l2_normalize(&[]), None);
        assert_eq!(l2_normalize(&[f32::NAN]), None);
        assert!(l2_normalize(&[3.0, 4.0]).is_some());
    }

    // ── query parsing ──

    #[test]
    fn query_accepts_json_array() {
        assert_eq!(
            parse_query_vec("[1.0, 2.0, 3.0]").unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn query_accepts_hex_blob() {
        let v = vec![1.0f32, -2.0];
        let hex = hex_encode(&pack_f32_le(&v));
        assert_eq!(parse_query_vec(&hex).unwrap(), v);
    }

    #[test]
    fn query_accepts_duckdb_binary_string() {
        // to_json(blob) emits "\\xNN" escapes + printable ASCII; 4 bytes = one f32
        assert_eq!(parse_query_vec("\\x00\\x00\\x80?").unwrap(), vec![1.0]); // 0x3F = '?'
        assert_eq!(parse_query_vec("\\x00\\x00\\x80\\x3F").unwrap(), vec![1.0]); // upper-case hex
                                                                                 // two f32s: [1.0, 0.0]
        assert_eq!(
            parse_query_vec("\\x00\\x00\\x80?\\x00\\x00\\x00\\x00").unwrap(),
            vec![1.0, 0.0]
        );
    }

    #[test]
    fn duckdb_binary_decode_escapes() {
        assert_eq!(
            decode_duckdb_binary("\\x00\\x00\\x80?").unwrap(),
            vec![0, 0, 128, b'?']
        );
        assert_eq!(
            decode_duckdb_binary("a\\x01b").unwrap(),
            vec![b'a', 1, b'b']
        );
        assert_eq!(decode_duckdb_binary("\\\\\\\"").unwrap(), vec![b'\\', b'"']);
        assert!(decode_duckdb_binary("\\x0").is_none()); // truncated escape
        assert!(decode_duckdb_binary("\\xzz").is_none()); // bad hex digit
    }

    #[test]
    fn blob_from_string_dispatches() {
        // hex form
        assert_eq!(
            blob_from_string("0000803F").unwrap(),
            vec![0, 0, 0x80, 0x3F]
        );
        // duckdb binary-string form (contains backslash)
        assert_eq!(
            blob_from_string("\\x00\\x00\\x80?").unwrap(),
            vec![0, 0, 0x80, b'?']
        );
        // raw printable bytes (non-hex)
        assert_eq!(blob_from_string("ab?").unwrap(), vec![b'a', b'b', b'?']);
        // even-length all-hex printable = treated as hex (documented ambiguity)
        assert_eq!(blob_from_string("abcd").unwrap(), vec![0xab, 0xcd]);
    }

    #[test]
    fn query_rejects_empty_and_malformed() {
        assert!(parse_query_vec("").is_err());
        assert!(parse_query_vec("[]").is_err());
        assert!(parse_query_vec("[1,2").is_err());
        assert!(parse_query_vec("zz").is_err());
        assert!(parse_query_vec("0000ff").is_err()); // len % 4 != 0 after decode
    }

    fn hex_encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    // ── similarity scan ──

    fn cand_pair(id: u64, emb: &[f32]) -> Value {
        serde_json::json!([id, emb])
    }

    #[test]
    fn scan_ranks_by_cosine_desc() {
        // query = [1,0]; [1,0] → 1.0, [0.9,0.1] → ~0.99, [0,1] → 0.0
        let cands =
            serde_json::json!([[1, [0.0, 1.0]], [2, [1.0, 0.0]], [3, [0.9, 0.1]],]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let r = v["results"].as_array().unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0]["row_id"], 2); // exact match first
        assert_eq!(r[1]["row_id"], 3);
        assert_eq!(r[2]["row_id"], 1);
        assert_eq!(r[0]["score"], 1.0);
        assert_eq!(v["scanned"], 3);
    }

    #[test]
    fn scan_threshold_filters() {
        let cands =
            serde_json::json!([[1, [1.0, 0.0]], [2, [0.5, 0.5]], [3, [0.0, 1.0]],]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.9).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let r = v["results"].as_array().unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0]["row_id"], 1);
    }

    #[test]
    fn scan_k_caps_and_sorts() {
        let cands = serde_json::json!([
            [1, [1.0, 0.0]],
            [2, [0.9, 0.1]],
            [3, [0.8, 0.2]],
            [4, [0.7, 0.3]],
            [5, [0.0, 1.0]],
        ])
        .to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 2, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let r = v["results"].as_array().unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0]["row_id"], 1);
        assert_eq!(r[1]["row_id"], 2);
    }

    #[test]
    fn scan_k_capped_at_10000() {
        // k=99999 behaves like k=10000 (no panic, bounded)
        let cands = serde_json::json!([[1, [1.0, 0.0]]]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 99_999, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn scan_skips_bad_len_blobs() {
        // hex "0000" = 2 bytes → len%4 != 0 → skipped
        let cands = serde_json::json!([[1, "0000"], [2, [1.0, 0.0]],]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 1);
        assert_eq!(v["results"][0]["row_id"], 2);
        assert_eq!(v["skipped_bad_len"], 1);
    }

    #[test]
    fn scan_dim_mismatch_scores_zero_and_counts() {
        let cands = serde_json::json!([
            [1, [1.0, 0.0, 5.0]], // dim 3 vs query dim 2
            [2, [1.0, 0.0]],
        ])
        .to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["skipped_dim"], 1);
        // dim-mismatch row present with score 0.0 (spec: score 0 + warning)
        assert_eq!(v["results"].as_array().unwrap().len(), 2);
        assert_eq!(v["results"][1]["row_id"], 1);
        assert_eq!(v["results"][1]["score"], 0.0);
    }

    #[test]
    fn scan_skips_null_embeddings() {
        let cands = serde_json::json!([[1, null], [2, [1.0, 0.0]],]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 1);
        assert_eq!(v["skipped_null"], 1);
    }

    #[test]
    fn scan_accepts_struct_form() {
        let cands = serde_json::json!([
            {"row_id": 10, "embeds": [1.0, 0.0]},
            {"row_id": 20, "embeds": [0.0, 1.0]},
        ])
        .to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let r = v["results"].as_array().unwrap();
        assert_eq!(r[0]["row_id"], 10);
    }

    #[test]
    fn scan_accepts_struct_hex_blob() {
        let emb = [1.0f32, 0.0];
        let hex = hex_encode(&pack_f32_le(&emb));
        let cands = serde_json::json!([{"row_id": 7, "embeds": hex}]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"][0]["row_id"], 7);
        assert_eq!(v["results"][0]["score"], 1.0);
    }

    #[test]
    fn scan_accepts_duckdb_binary_string() {
        // what to_json(blob) emits: "\\x00\\x00\\x80?\\x00\\x00\\x00\\x00" = f32 LE [1.0, 0.0]
        let cands = serde_json::json!([
            {"row_id": 1, "embeds": "\\x00\\x00\\x80?\\x00\\x00\\x00\\x00"},
            {"row_id": 2, "embeds": "\\x00\\x00\\x00\\x00\\x00\\x00\\x80?"},
        ])
        .to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let r = v["results"].as_array().unwrap();
        assert_eq!(r[0]["row_id"], 1);
        assert_eq!(r[0]["score"], 1.0);
        assert_eq!(r[1]["row_id"], 2);
        assert_eq!(r[1]["score"], 0.0);
    }

    #[test]
    fn scan_empty_candidates_returns_empty() {
        let out = similarity_from_json("[1.0, 0.0]", "[]", 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 0);
        assert_eq!(v["scanned"], 0);

        let out = similarity_from_json("[1.0, 0.0]", "", 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn scan_malformed_candidates_error() {
        assert!(similarity_from_json("[1.0, 0.0]", "not json", 10, 0.0).is_err());
        assert!(similarity_from_json("[1.0, 0.0]", "{\"a\":1}", 10, 0.0).is_err());
        assert!(similarity_from_json("", "[]", 10, 0.0).is_err());
        assert!(similarity_from_json("[1.0, 0.0]", "[]", 0, 0.0).is_err());
    }

    #[test]
    fn scan_malformed_element_counted_not_fatal() {
        let cands = serde_json::json!(["garbage", [1, [1.0, 0.0]],]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 1);
        assert_eq!(v["skipped_bad_len"], 1);
    }

    #[test]
    fn scan_numeric_row_id_tie_break() {
        // Equal scores → smaller numeric row_id first ("10" < "9" string compare
        // would be wrong; numeric compare is right).
        let cands = serde_json::json!([[10, [1.0, 0.0]], [9, [1.0, 0.0]],]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let r = v["results"].as_array().unwrap();
        assert_eq!(r[0]["row_id"], 9);
        assert_eq!(r[1]["row_id"], 10);
    }

    #[test]
    fn scan_cancel_returns_partial() {
        // > CANCEL_CHECK_INTERVAL rows so the scan hits an interval boundary.
        let cands: Vec<Value> = (0..5000)
            .map(|i| serde_json::json!([i as u64, [1.0, 0.0]]))
            .collect();
        let cands = serde_json::json!(cands).to_string();
        set_similarity_cancel(true);
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        set_similarity_cancel(false);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["cancelled"], true);
        assert_eq!(v["scanned"], 4096); // rows 0..4095 processed, cancel at 4096
        assert_eq!(v["results"].as_array().unwrap().len(), 10);
    }

    #[test]
    fn scan_cancel_reset_allows_full() {
        set_similarity_cancel(false);
        let cands = serde_json::json!([[1, [1.0, 0.0]]]).to_string();
        let out = similarity_from_json("[1.0, 0.0]", &cands, 10, 0.0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["cancelled"], false);
        assert_eq!(v["results"].as_array().unwrap().len(), 1);
    }
}
