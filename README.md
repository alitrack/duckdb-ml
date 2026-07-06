# duckdb-ml

Lightweight, columnar-native, train+inference DuckDB machine learning extension. Pure Rust, MIT license.

## Install

```sql
INSTALL ml FROM community;
LOAD ml;
```

## Usage

### Training (Planned for v0.1)

```sql
-- Train a linear regression model
CREATE MODEL house_price
USING linear_regression
FEATURES (sqft, bedrooms, bathrooms)
TARGET price
FROM houses;
```

### Prediction

```sql
-- Predict using literal feature values
SELECT * FROM ml_predict('house_price', 2200, 3, 2);
-- returns: prediction = 485000.0
```

### Model Management

```sql
-- List all trained models
SELECT * FROM ml_list_models();
-- name | algorithm | created_at | status | r_squared | mse

-- Model metadata (Planned)
DESCRIBE MODEL house_price;

-- Delete a model (Planned)
DROP MODEL house_price;
```

## Supported Algorithms (v0.1)

| Algorithm | Type | Status |
|-----------|------|--------|
| Linear Regression | Regression | ✅ |
| Ridge Regression | Regression | ✅ |
| Logistic Regression | Classification | ✅ |

## Build from Source

```bash
git clone https://github.com/alitrack/duckdb-ml.git
cd duckdb-ml
cargo build --release
# produces target/release/libduckdb_ml.so
```

## License

MIT
