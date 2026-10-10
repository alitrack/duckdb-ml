---
name: duckdb-ml-vector
description: >
  Embeddings, similarity search, and association rules with the ml DuckDB
  extension: ml_embed (ONNX encoders → f32 blobs), ml_similarity_value
  (full-scan cosine Top-K), and ml_assoc_rules (Apriori market-basket).
  Use for semantic search over columns, dedup by embedding, or mining
  transaction co-occurrence — all in SQL.
version: 0.2.0
---

# ml: embeddings, similarity & association rules

## Embeddings (ONNX encoder)

```sql
-- Encoder registered by name (e.g. 'clip'); features_json in, f32 LE blob per row
UPDATE media SET embeds = ml_embed('clip', features_json);
```

Embeddings land as BLOBs in-column — pair with similarity below, or export
for external indexes.

## Similarity Top-K (full-scan cosine)

```sql
SELECT ml_similarity_value(
    '[0.1, 0.2, ...]',                                                    -- query vector
    (SELECT to_json(list({'row_id': id, 'embeds': embeds})) FROM media),  -- corpus
    10, 0.3);                                                             -- top-k, threshold
```

Full-scan — exact, no training, no index build. Right answer up to ~100k
rows; beyond that use the `turbovec` extension (TurboQuant compressed ANN).

## Association rules (Apriori)

```sql
SELECT ml_assoc_rules(
    (SELECT to_json(list({'tid': txn_id, 'items': items}))
     FROM (SELECT txn_id, list(item_id ORDER BY item_id) AS items
           FROM orders GROUP BY txn_id) t),
    0.05,  -- min_support (fraction)
    0.6);  -- min_confidence (fraction)
```

Input is one JSON array of `{tid, items}` per transaction; items must be
sorted within each transaction for deterministic output.

## Pattern — semantic search end to end

```sql
-- 1. embed once
UPDATE docs SET emb = ml_embed('clip', content_features);
-- 2. search by example
SELECT ml_similarity_value(
    (SELECT emb FROM docs WHERE id = 42),
    (SELECT to_json(list({'row_id': id, 'embeds': emb})) FROM docs),
    10, 0.3);
```
