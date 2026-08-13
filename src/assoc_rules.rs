//! Association rules mining (Apriori) — market basket analysis.
//!
//! SQL surface (scalar, JSON collected via subquery, same family as
//! `ml_ols` / `ml_similarity_value`):
//!
//! ```sql
//! SELECT ml_assoc_rules(
//!     (SELECT to_json(list({'tid': txn_id, 'items': items}))
//!      FROM (SELECT txn_id, list(item_id ORDER BY item_id) AS items
//!            FROM orders GROUP BY txn_id) t),
//!     0.05,   -- min_support (fraction, (0, 1])
//!     0.6);   -- min_confidence (fraction, (0, 1])
//! ```
//!
//! Input: `[{"tid": 1, "items": ["milk","bread"]}, ...]` — items accept
//! string/number/bool scalars (deduplicated per transaction); `tid` optional.
//!
//! Output JSON: `{"rules":[{antecedent,consequent,support,confidence,lift}],
//! "frequent_itemsets":[{items,support}], "stats":{transactions,candidates,
//! rules,cancelled,truncated}}`.
//!
//! Cancellation: a global atomic flag (set via [`set_assoc_cancel`], for
//! embedded/rlib callers) aborts mid-scan and returns partial results with
//! `"cancelled": true`. Checks start at row > 0 every 4096 rows so small
//! scans are immune (parallel-test isolation, same pattern as embed.rs).

use arrow::array::{Array, ArrayRef, Float64Array, StringArray, UInt64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use duckdb::vscalar::arrow::{ArrowFunctionSignature, VArrowScalar};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Max candidate itemsets per round before truncating (low-support blowup guard).
const MAX_CANDIDATES: usize = 2_000_000;
/// Max rules emitted before truncating.
const MAX_RULES: usize = 1_000_000;
/// Cancel check frequency in transaction-counting loops.
const CANCEL_CHECK_INTERVAL: u64 = 4096;

/// Global cancellation flag for in-flight Apriori scans (embedded use).
static ASSOC_CANCEL: AtomicBool = AtomicBool::new(false);

/// Set/reset the global association-mining cancellation flag.
pub fn set_assoc_cancel(cancel: bool) {
    ASSOC_CANCEL.store(cancel, AtomicOrdering::Relaxed);
}

/// True when the caller requested cancellation and enough work has passed the
/// check interval — small scans (< `CANCEL_CHECK_INTERVAL` work units) are
/// never interrupted, which keeps parallel tests isolated from each other.
fn cancel_check(work: &mut u64) -> bool {
    *work += 1;
    work.is_multiple_of(CANCEL_CHECK_INTERVAL) && ASSOC_CANCEL.load(AtomicOrdering::Relaxed)
}

/// One mined association rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub antecedent: Vec<String>,
    pub consequent: Vec<String>,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
}

/// Aggregated mining result.
#[derive(Debug, Clone, PartialEq)]
pub struct AssocResult {
    pub rules: Vec<Rule>,
    pub frequent_itemsets: Vec<(Vec<String>, u64)>,
    pub transactions: u64,
    pub candidates: u64,
    pub cancelled: bool,
    pub truncated: bool,
}

/// Normalize one item JSON value to a String key; `None` for null.
/// Objects/arrays are rejected (malformed input).
fn item_key(v: &Value) -> Result<Option<String>, String> {
    match v {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        Value::Number(n) => Ok(Some(n.to_string())),
        Value::Bool(b) => Ok(Some(b.to_string())),
        _ => Err("items must contain scalars (string/number/bool), found object/array".into()),
    }
}

/// Parse transactions JSON: `[{"tid":N,"items":[...]}, ...]` → sorted,
/// deduplicated item lists. `tid` is optional (index used when absent).
pub fn parse_transactions(json_text: &str) -> Result<Vec<Vec<String>>, String> {
    let v: Value = serde_json::from_str(json_text)
        .map_err(|e| format!("transactions JSON is malformed: {e}"))?;
    let arr = v
        .as_array()
        .ok_or("transactions JSON must be an array of {tid, items} objects")?;

    let mut out = Vec::with_capacity(arr.len());
    for (i, txn) in arr.iter().enumerate() {
        let obj = txn
            .as_object()
            .ok_or_else(|| format!("transaction #{i} must be an object"))?;
        let items_val = obj
            .get("items")
            .ok_or_else(|| format!("transaction #{i} is missing \"items\""))?;
        let items_arr = items_val
            .as_array()
            .ok_or_else(|| format!("transaction #{i} \"items\" must be an array"))?;
        let mut set: HashSet<String> = HashSet::with_capacity(items_arr.len());
        for (j, item) in items_arr.iter().enumerate() {
            match item_key(item) {
                Ok(Some(k)) => {
                    set.insert(k);
                }
                Ok(None) => {} // null item: skip
                Err(e) => return Err(format!("transaction #{i} item #{j}: {e}")),
            }
        }
        let mut items: Vec<String> = set.into_iter().collect();
        items.sort_unstable();
        if !items.is_empty() {
            out.push(items);
        }
    }
    Ok(out)
}

/// Apriori: mine frequent itemsets with support ≥ min_support_abs.
///
/// Uses an inverted index (item → transaction indices) so candidate support
/// is computed as the intersection of its items' transaction lists.
/// Returns `(frequent, candidates_checked, cancelled)`.
fn apriori(
    transactions: &[Vec<String>],
    min_support_abs: u64,
    max_itemset_size: usize,
    truncated: &mut bool,
) -> (Vec<(Vec<String>, u64)>, u64, bool) {
    let mut candidates_checked: u64 = 0;
    let mut work: u64 = 0;
    let mut cancelled = false;

    // L1: single-item counts via inverted index.
    let mut item_txns: HashMap<String, Vec<usize>> = HashMap::new();
    for (t, txn) in transactions.iter().enumerate() {
        for item in txn {
            item_txns.entry(item.clone()).or_default().push(t);
        }
    }
    let mut frequent: Vec<(Vec<String>, u64)> = item_txns
        .iter()
        .filter(|(_, idx)| idx.len() as u64 >= min_support_abs)
        .map(|(item, idx)| (vec![item.clone()], idx.len() as u64))
        .collect();
    frequent.sort();

    let mut level: Vec<Vec<String>> = frequent.iter().map(|(i, _)| i.clone()).collect();
    let mut k = 1usize;
    while !level.is_empty() && (max_itemset_size == 0 || k < max_itemset_size) {
        // Generate C_{k+1}: join + prune.
        let mut candidates: Vec<Vec<String>> = Vec::new();
        for i in 0..level.len() {
            if cancel_check(&mut work) {
                cancelled = true;
                return (frequent, candidates_checked, cancelled);
            }
            let a = &level[i];
            for b in &level[i + 1..] {
                // Join: first k-1 items equal, last differs.
                if a[..k - 1] != b[..k - 1] {
                    continue;
                }
                let mut cand = a.clone();
                cand.push(b[k - 1].clone());
                // Prune: all k-subsets must be frequent.
                let mut ok = true;
                for drop in 0..cand.len() {
                    let mut sub = cand.clone();
                    sub.remove(drop);
                    if !level.binary_search(&sub).is_ok() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    candidates.push(cand);
                    if candidates.len() >= MAX_CANDIDATES {
                        *truncated = true;
                        return (frequent, candidates_checked, cancelled);
                    }
                }
            }
        }
        if candidates.is_empty() {
            break;
        }
        candidates_checked += candidates.len() as u64;

        // Support counting via inverted-index intersection.
        let mut counts: Vec<(Vec<String>, u64)> = Vec::with_capacity(candidates.len());
        for cand in &candidates {
            if cancel_check(&mut work) {
                cancelled = true;
                return (frequent, candidates_checked, cancelled);
            }
            // Start from the shortest item's transaction list.
            let mut base: Option<Vec<usize>> = None;
            for item in cand {
                let idx = &item_txns[item];
                if base.is_none() || idx.len() < base.as_ref().unwrap().len() {
                    base = Some(idx.clone());
                }
            }
            let base = base.unwrap_or_default();
            let mut cnt = 0u64;
            for &t in &base {
                let txn = &transactions[t];
                if cand.iter().all(|item| txn.binary_search(item).is_ok()) {
                    cnt += 1;
                }
            }
            if cnt >= min_support_abs {
                counts.push((cand.clone(), cnt));
            }
        }
        counts.sort();
        frequent.extend(counts.iter().cloned());
        level = counts.into_iter().map(|(items, _)| items).collect();
        k += 1;
    }
    (frequent, candidates_checked, cancelled)
}

/// Generate rules from frequent itemsets; antecedent = any non-empty proper
/// subset, consequent = remainder. `conf = supp(F)/supp(A)`,
/// `lift = conf / (supp(B)/n)`. Returns `(rules, cancelled)`.
fn generate_rules(
    frequent: &[(Vec<String>, u64)],
    min_confidence: f64,
    n: u64,
    truncated: &mut bool,
) -> (Vec<Rule>, bool) {
    let supp: HashMap<&[String], u64> = frequent
        .iter()
        .map(|(items, c)| (items.as_slice(), *c))
        .collect();
    let mut rules = Vec::new();
    let mut work: u64 = 0;
    let mut cancelled = false;

    for (itemset, count_f) in frequent {
        if itemset.len() < 2 {
            continue;
        }
        let m = itemset.len();
        let full_mask = 1usize << m;
        for mask in 1..full_mask - 1 {
            if cancel_check(&mut work) {
                cancelled = true;
                return (rules, cancelled);
            }
            let mut ant: Vec<String> = Vec::new();
            let mut cons: Vec<String> = Vec::new();
            for (i, item) in itemset.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    ant.push(item.clone());
                } else {
                    cons.push(item.clone());
                }
            }
            let Some(&count_a) = supp.get(ant.as_slice()) else {
                continue; // Apriori property guarantees existence; defensive
            };
            let confidence = *count_f as f64 / count_a as f64;
            if confidence < min_confidence {
                continue;
            }
            let Some(&count_b) = supp.get(cons.as_slice()) else {
                continue;
            };
            let lift = confidence / (count_b as f64 / n as f64);
            let support = *count_f as f64 / n as f64;
            rules.push(Rule {
                antecedent: ant,
                consequent: cons,
                support,
                confidence,
                lift,
            });
            if rules.len() >= MAX_RULES {
                *truncated = true;
                return (rules, cancelled);
            }
        }
    }
    (rules, cancelled)
}

/// Deterministic rule ordering: confidence desc → support desc → ant len asc.
fn order_rules(rules: &mut [Rule]) {
    rules.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.support.partial_cmp(&a.support).unwrap_or(Ordering::Equal))
            .then_with(|| a.antecedent.len().cmp(&b.antecedent.len()))
    });
}

/// Full pipeline: parse → apriori → rules → JSON. Public for embedded use and
/// tests; errors are user-facing messages.
pub fn assoc_rules_from_json(
    transactions_json: &str,
    min_support: f64,
    min_confidence: f64,
    max_itemset_size: usize,
) -> Result<String, String> {
    if !(min_support > 0.0 && min_support <= 1.0) {
        return Err("min_support must be in (0, 1]".into());
    }
    if !(min_confidence > 0.0 && min_confidence <= 1.0) {
        return Err("min_confidence must be in (0, 1]".into());
    }
    let transactions = parse_transactions(transactions_json)?;
    let n = transactions.len() as u64;
    let mut truncated = false;

    if n == 0 {
        return Ok(json!({
            "rules": [],
            "frequent_itemsets": [],
            "stats": {"transactions": 0, "candidates": 0, "rules": 0,
                      "cancelled": false, "truncated": false}
        })
        .to_string());
    }

    let min_support_abs = (min_support * n as f64).ceil() as u64;
    let (frequent, candidates, apriori_cancelled) = apriori(
        &transactions,
        min_support_abs,
        max_itemset_size,
        &mut truncated,
    );
    if frequent.is_empty() {
        return Ok(json!({
            "rules": [],
            "frequent_itemsets": [],
            "stats": {"transactions": n, "candidates": candidates, "rules": 0,
                      "cancelled": apriori_cancelled, "truncated": truncated}
        })
        .to_string());
    }

    let (mut rules, rules_cancelled) = generate_rules(&frequent, min_confidence, n, &mut truncated);
    order_rules(&mut rules);
    let cancelled = apriori_cancelled || rules_cancelled;

    let itemsets_json: Vec<Value> = frequent
        .iter()
        .map(|(items, count)| json!({"items": items, "support": *count as f64 / n as f64}))
        .collect();
    let rules_json: Vec<Value> = rules
        .iter()
        .map(|r| {
            json!({
                "antecedent": r.antecedent,
                "consequent": r.consequent,
                "support": r.support,
                "confidence": r.confidence,
                "lift": r.lift,
            })
        })
        .collect();

    Ok(json!({
        "rules": rules_json,
        "frequent_itemsets": itemsets_json,
        "stats": {
            "transactions": n,
            "candidates": candidates,
            "rules": rules.len(),
            "cancelled": cancelled,
            "truncated": truncated,
        }
    })
    .to_string())
}

// ── Shared scalar helpers (same shape as embed.rs) ──

fn col_str(input: &RecordBatch, idx: usize) -> Result<&StringArray, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("expected VARCHAR column".into())
}

fn col_f64(input: &RecordBatch, idx: usize) -> Result<&Float64Array, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or("expected DOUBLE column".into())
}

fn col_u64(input: &RecordBatch, idx: usize) -> Result<&UInt64Array, Box<dyn Error>> {
    input
        .column(idx)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or("expected UBIGINT column".into())
}

// ── ml_assoc_rules(transactions, min_support, min_confidence [, max_itemset_size]) ──

/// Apriori association rules; returns the JSON payload (see
/// [`assoc_rules_from_json`]).
pub struct AssocRulesFn;

impl VArrowScalar for AssocRulesFn {
    type State = ();

    fn invoke(_state: &Self::State, input: RecordBatch) -> Result<ArrayRef, Box<dyn Error>> {
        let n = input.num_rows();
        let n_args = input.num_columns();
        if !(3..=4).contains(&n_args) {
            return Err(
                "ml_assoc_rules requires 3..4 args: transactions, min_support, min_confidence[, max_itemset_size]"
                    .into(),
            );
        }

        let txns_col = col_str(&input, 0)?;
        let support_col = col_f64(&input, 1)?;
        let confidence_col = col_f64(&input, 2)?;

        let max_itemset_size: usize = if n_args >= 4 {
            let col = col_u64(&input, 3)?;
            if col.is_null(0) {
                0
            } else {
                col.value(0) as usize
            }
        } else {
            0
        };

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if txns_col.is_null(i) || support_col.is_null(i) || confidence_col.is_null(i) {
                out.push(None);
                continue;
            }
            let json = assoc_rules_from_json(
                txns_col.value(i),
                support_col.value(i),
                confidence_col.value(i),
                max_itemset_size,
            )?;
            out.push(Some(json));
        }
        let arr = StringArray::from(out);
        Ok(Arc::new(arr))
    }

    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![
            ArrowFunctionSignature::exact(
                vec![DataType::Utf8, DataType::Float64, DataType::Float64],
                DataType::Utf8,
            ),
            ArrowFunctionSignature::exact(
                vec![
                    DataType::Utf8,
                    DataType::Float64,
                    DataType::Float64,
                    DataType::UInt64,
                ],
                DataType::Utf8,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parsing ──

    #[test]
    fn parse_accepts_mixed_scalars_and_dedupes() {
        let txns =
            parse_transactions(r#"[{"tid":1,"items":["milk","bread","milk",42,42.5,true,null]}]"#)
                .unwrap();
        assert_eq!(
            txns,
            vec![vec![
                "42".to_string(),
                "42.5".to_string(),
                "bread".to_string(),
                "milk".to_string(),
                "true".to_string()
            ]]
        );
    }

    #[test]
    fn parse_skips_empty_transactions() {
        let txns =
            parse_transactions(r#"[{"tid":1,"items":[]},{"tid":2,"items":[null]}]"#).unwrap();
        assert!(txns.is_empty());
    }

    #[test]
    fn parse_rejects_object_item() {
        let err = parse_transactions(r#"[{"tid":1,"items":[{"x":1}]}]"#).unwrap_err();
        assert!(err.contains("object/array"), "{err}");
    }

    #[test]
    fn parse_rejects_non_array_root() {
        assert!(parse_transactions(r#"{"tid":1}"#).is_err());
    }

    #[test]
    fn parse_missing_items_errors() {
        assert!(parse_transactions(r#"[{"tid":1}]"#).is_err());
    }

    // ── apriori ──

    /// Classic market basket: milk/bread/diaper example (5 transactions).
    /// milk:4, bread:5, diaper:3, beer:1; {milk,bread}:4, {milk,diaper}:3,
    /// {bread,diaper}:3, {milk,bread,diaper}:3.
    fn classic_transactions() -> Vec<Vec<String>> {
        parse_transactions(
            r#"[
                {"tid":1,"items":["milk","bread","diaper"]},
                {"tid":2,"items":["milk","bread","diaper"]},
                {"tid":3,"items":["milk","bread","diaper"]},
                {"tid":4,"items":["milk","bread"]},
                {"tid":5,"items":["bread","beer"]}
            ]"#,
        )
        .unwrap()
    }

    /// Same data as [`classic_transactions`] but as the raw JSON string.
    const CLASSIC_JSON: &str = r#"[
        {"tid":1,"items":["milk","bread","diaper"]},
        {"tid":2,"items":["milk","bread","diaper"]},
        {"tid":3,"items":["milk","bread","diaper"]},
        {"tid":4,"items":["milk","bread"]},
        {"tid":5,"items":["bread","beer"]}
    ]"#;

    #[test]
    fn freq_itemsets_basic() {
        let txns = classic_transactions();
        let (freq, _, _) = apriori(&txns, 2, 0, &mut false); // min_support = 2/5 = 0.4
        let map: HashMap<&[String], u64> = freq.iter().map(|(i, c)| (i.as_slice(), *c)).collect();
        assert_eq!(map.get(&["milk".to_string()][..]), Some(&4));
        assert_eq!(map.get(&["bread".to_string()][..]), Some(&5));
        assert_eq!(map.get(&["diaper".to_string()][..]), Some(&3));
        // itemsets are stored in lexicographic order
        assert_eq!(
            map.get(&["bread".to_string(), "milk".to_string()][..]),
            Some(&4)
        );
        assert_eq!(
            map.get(&["milk".to_string(), "bread".to_string()][..]),
            None
        );
        assert_eq!(freq.len(), 7);
    }

    #[test]
    fn freq_itemsets_respects_support() {
        let txns = classic_transactions();
        let (freq, _, _) = apriori(&txns, 4, 0, &mut false); // support ≥ 4/5 = 0.8
        assert_eq!(
            freq,
            vec![
                (vec!["bread".to_string()], 5),
                (vec!["milk".to_string()], 4),
                (vec!["bread".to_string(), "milk".to_string()], 4),
            ]
        );
    }

    #[test]
    fn itemset_size_cap_limits_level() {
        let txns = classic_transactions();
        let (freq, _, _) = apriori(&txns, 2, 2, &mut false); // max size 2
        assert!(
            freq.iter().all(|(items, _)| items.len() <= 2),
            "got {freq:?}"
        );
        assert!(freq.len() >= 6);
    }

    // ── rules ──

    #[test]
    fn rules_support_conf_lift() {
        let txns = classic_transactions();
        let (freq, _, _) = apriori(&txns, 2, 0, &mut false);
        let mut rules = generate_rules(&freq, 0.0, txns.len() as u64, &mut false).0;
        order_rules(&mut rules);
        // {milk} → {bread}: supp({milk,bread})=4/5=0.8, conf=4/4=1.0,
        // lift = conf / supp(bread) = 1.0/1.0 = 1.0
        let r = rules
            .iter()
            .find(|r| r.antecedent == vec!["milk"] && r.consequent == vec!["bread"])
            .expect("rule exists");
        assert!((r.support - 0.8).abs() < 1e-9);
        assert!((r.confidence - 1.0).abs() < 1e-9);
        assert!((r.lift - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rule_filter_confidence() {
        let txns = classic_transactions();
        let (freq, _, _) = apriori(&txns, 2, 0, &mut false);
        let rules = generate_rules(&freq, 0.9, txns.len() as u64, &mut false).0;
        // {bread} → {milk}: supp({bread,milk})=4, supp({bread})=5 → conf=0.8 < 0.9 filtered
        assert!(rules.iter().all(|r| r.confidence >= 0.9 - 1e-9
            && !(r.antecedent == vec!["bread"] && r.consequent == vec!["milk"])));
        // {diaper} → {milk}: conf = 3/3 = 1.0 ≥ 0.9 present
        assert!(rules
            .iter()
            .any(|r| r.antecedent == vec!["diaper"] && r.consequent == vec!["milk"]));
    }

    #[test]
    fn rules_require_size_two_itemsets() {
        let txns = parse_transactions(r#"[{"tid":1,"items":["a","b"]}]"#).unwrap();
        let (freq, _, _) = apriori(&txns, 1, 0, &mut false);
        // only {a},{b},{a,b} — {a,b} size 2 → rules {a}→{b},{b}→{a}
        let rules = generate_rules(&freq, 0.0, 1, &mut false).0;
        assert_eq!(rules.len(), 2);
    }

    #[test]
    fn deterministic_order() {
        let a = assoc_rules_from_json(CLASSIC_JSON, 0.4, 0.5, 0).unwrap();
        let b = assoc_rules_from_json(CLASSIC_JSON, 0.4, 0.5, 0).unwrap();
        assert_eq!(a, b);
    }

    // ── boundaries & validation ──

    #[test]
    fn param_validation() {
        assert!(assoc_rules_from_json("[]", 0.0, 0.5, 0).is_err());
        assert!(assoc_rules_from_json("[]", 1.5, 0.5, 0).is_err());
        assert!(assoc_rules_from_json("[]", 0.5, 0.0, 0).is_err());
        assert!(assoc_rules_from_json("[]", 0.5, 1.1, 0).is_err());
    }

    #[test]
    fn empty_and_single_transactions() {
        let out = assoc_rules_from_json("[]", 0.5, 0.5, 0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["rules"].as_array().unwrap().len(), 0);
        assert_eq!(v["stats"]["transactions"], 0);

        let out = assoc_rules_from_json(r#"[{"tid":1,"items":["a"]}]"#, 0.5, 0.5, 0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["rules"].as_array().unwrap().len(), 0);
        assert_eq!(v["frequent_itemsets"].as_array().unwrap().len(), 1);
    }

    // ── cancel & caps ──

    #[test]
    fn cancel_returns_partial() {
        // 5000 transactions over a 100-item pool: the L1 join/count alone
        // drives the work counter past CANCEL_CHECK_INTERVAL, so the global
        // flag is observed mid-scan and partial results come back.
        let mut parts = Vec::with_capacity(5000);
        for t in 0..5000usize {
            let items: Vec<String> = (0..5).map(|d| format!("i{}", (t + d) % 100)).collect();
            parts.push(format!(
                r#"{{"tid":{},"items":["{}"]}}"#,
                t,
                items.join("\",\"")
            ));
        }
        let json = format!("[{}]", parts.join(","));
        set_assoc_cancel(true);
        let out = assoc_rules_from_json(&json, 0.04, 0.5, 0).unwrap();
        set_assoc_cancel(false);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["stats"]["cancelled"], true);
        // partial: rules may be empty but the call must not fail
        assert!(v["rules"].is_array());
    }

    #[test]
    fn small_scan_not_cancelled() {
        // With the flag clear, a small scan is never interrupted and reports
        // cancelled=false. Parallel-safe: even when another test sets the
        // flag, scans under CANCEL_CHECK_INTERVAL work units are immune.
        let out = assoc_rules_from_json(CLASSIC_JSON, 0.4, 0.5, 0).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["stats"]["cancelled"], false);
        assert_eq!(v["stats"]["truncated"], false);
    }
}
