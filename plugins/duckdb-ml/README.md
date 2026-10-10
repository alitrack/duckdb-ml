# duckdb-ml — Claude Code Plugin

Two skills that give your AI assistant verified knowledge of the
[duckdb-ml](https://github.com/alitrack/duckdb-ml) extension: the 34-algorithm
train/predict/deploy lifecycle, and embeddings/similarity/association rules.
Zero Python — everything runs inside DuckDB.

## Install

```text
/plugin marketplace add alitrack/duckdb-ml
/plugin install duckdb-ml@duckdb-ml
```

For development against this repo: `claude --plugin-dir ./plugins/duckdb-ml`.

## Skills

| Skill | Covers |
|---|---|
| `duckdb-ml-train` | ml_train/predict/batch, algorithm picker, deploy/rollback, snapshots |
| `duckdb-ml-vector` | ml_embed (ONNX), ml_similarity_value Top-K, ml_assoc_rules (Apriori) |

## Prerequisite

```sql
INSTALL ml FROM community;
LOAD ml;
```

MIT, alitrack. Skills live in [`plugins/duckdb-ml/skills/`](plugins/duckdb-ml/skills/)
and are kept in sync with the extension README at each release.
