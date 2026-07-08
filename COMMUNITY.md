# DuckDB Community Extension Submission

## Checklist

- [x] Extension builds clean (`cargo build --release`)
- [x] Zero clippy warnings (`cargo clippy -- -D warnings`)
- [x] 36 tests passing (`cargo test --lib`)
- [x] `.duckdb_extension` packaged (`make release`)
- [x] Extension verified (`python3 scripts/verify_extension.py`)
- [x] MIT licensed
- [x] `README.md` with usage examples
- [ ] Platform builds: linux_amd64, osx_arm64, osx_amd64
- [ ] DuckDB extension repository PR

## Build

```bash
make release   # produces target/duckdb_ml.duckdb_extension
```

## Platform Matrix

| Platform | Build command | Output |
|----------|--------------|--------|
| `linux_amd64` | `cargo build --release` | `libduckdb_ml.so` |
| `osx_arm64` | `cargo build --release` (on Mac) | `libduckdb_ml.dylib` |
| `osx_amd64` | `cargo build --release` (on x86 Mac) | `libduckdb_ml.dylib` |

## Extension Metadata

- **Name:** `ml`
- **Description:** Train+inference machine learning in SQL (18 algorithms, zero Python)
- **Version:** v0.15.0
- **DuckDB min version:** v1.1.0
- **License:** MIT
- **Repository:** https://github.com/alitrack/duckdb-ml

## Submission Steps

1. Build for all platforms
2. Create PR to `duckdb/extension-ci-tools` adding this extension
3. Add `ml` to `extensions.csv` or extension configuration
4. Wait for CI build and review
