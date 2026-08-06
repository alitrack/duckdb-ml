

# duckdb-ml

**Extensión de ML ligera, nativa de columnas y para entrenamiento e inferencia en DuckDB.**
Sin dependencias de Python. 18 algoritmos en Rust puro.

```sql
-- Train
SELECT * FROM ml_train('my_model', 'random_forest', '[..]', '[[..]]', '{"n_estimators":100}');

-- Predict (single)
SELECT * FROM ml_predict('my_model', '[3.0, 4.5]');

-- Predict (batch)
SELECT * FROM ml_predict_batch('my_model', '[[1.0,2.0],[3.0,4.0]]');
SELECT * FROM ml_predict_batch('my_model', '@my_model'); -- re-use training data

-- AutoML: compare all regression algorithms
SELECT * FROM ml_compare('exp', '[...]', '[[..]]', '[]', 'regression');

-- Deploy + rollback
SELECT * FROM ml_deploy('my_model', 'best_score');
SELECT * FROM ml_deploy('my_model', 'rollback');

-- Data version tracking
SELECT * FROM ml_snapshot('my_model', 'train_data', 4, 250, 'target', '["x1","x2"]', 'abc123');
SELECT * FROM ml_list_snapshots('my_model');
```

## Algoritmos (18)

| Categoría | Algoritmos |
|----------|-----------|
| **Linear** | `linear_regression`, `ridge_regression`, `lasso_regression` |
| **Tree** | `decision_tree`, `random_forest` |
| **Gradient Boosting** | `xgboost_regression`, `xgboost_binary` (pure-Rust GBDT) |
| **Neural** | `mlp_regressor` (1-layer, ReLU, SGD+momentum) |
| **Distance** | `knn_regressor`, `knn_classifier` |
| **Bayesian** | `naive_bayes` |
| **Clustering** | `kmeans` |
| **Dim Reduction** | `pca` |
| **External** | `xgboost_regressor`, `xgboost_classifier` (load via `ml_load_xgboost`), `onnx` (load via `ml_load_onnx`) |

## Ejemplo de Flujo Completo

```sql
-- 1. Train with AutoML (compares linear, lasso, rf, knn, mlp)
SELECT * FROM ml_compare('house_exp', '[300000,450000,...]',
    '[[3,2,1500],[4,2,2200],...]', '[]', 'regression');

-- 2. Deploy best model
SELECT * FROM ml_deploy('house_exp', 'best_score');

-- 3. Batch predict on training data (auto-cached)
SELECT * FROM ml_predict_batch('house_exp', '@house_exp');

-- 4. Track data version
SELECT * FROM ml_snapshot('house_exp', 'houses_2025', 3, 500,
    'price', '["bedrooms","bathrooms","sqft"]',
    hash_training_data(features, targets));

-- 5. Register an external XGBoost model
SELECT * FROM ml_load_xgboost('xgb_model', '/path/to/model.json');

-- 6. Manual training with custom params
SELECT * FROM ml_train('custom_rf', 'random_forest',
    '[10,20,30,40]', '[[1,2],[3,4],[5,6],[7,8]]',
    '{"n_estimators":200,"max_depth":5}');

-- 7. Query model registry
SELECT * FROM ml_list_models;
```

## Características

- **Entrenamiento en SQL** — sin Python, sin Jupyter, sin procesos externos
- **AutoML** — `ml_compare` entrena todos los algoritmos en paralelo y devuelve una tabla de comparación
- **Gestión de versiones** — despliegue/rollback con estrategias (`best_score`, `most_recent`, `rollback`)
- **Predicción por lotes** — matrices JSON 2D o referencias de datos en caché `@model_name`
- **Linaje de datos** — `ml_snapshot` registra columnas de características, conteos de muestras y hashes de datos
- **Seguimiento de experimentos** — tablas nativas de DuckDB (`duckdb_ml.experiments`, `runs`, `metrics`, `params`)
- **XGBoost en Rust puro** — entrena ensambles GBDT y serializa a JSON compatible con XGBoost
- **Modelos externos** — carga archivos ONNX y JSON de XGBoost preentrenados
- **18 algoritmos** — lineales, árboles, boosting, neuronales, basados en distancia, bayesianos, agrupamiento y reducción de dimensionalidad

## Arquitectura

```mermaid
flowchart TD
    A[ml_train / ml_compare] --> B[Global ModelRegistry]
    B --> C[ml_predict / ml_predict_batch]
    A --> D[Dataset Cache]
    D --> C
    B --> E[ml_deploy / rollback]
    A --> F[ml_snapshot / ml_list_snapshots]
    G[ml_load_xgboost / ml_load_onnx] --> B
```

Todos los modelos residen en un registro global seguro para hilos (caché LRU, límite de 100 modelos).
El estado de despliegue, los metadatos de instantáneas y los conjuntos de datos en caché están todos en memoria
(restricción de extensión cargable de DuckDB: sin `Connection` en funciones de tabla).

## Compilación e Instalación

```bash
git clone git@github.com:alitrack/duckdb-ml.git
cd duckdb-ml
cargo build --release
```

Cargar en DuckDB:
```sql
LOAD '/path/to/libduckdb_ml.so';
SELECT duckdb_ml();
```

## Desarrollo

```bash
cargo test --lib     # 36 tests
cargo clippy -- -D warnings
cargo fmt
```

## Licencia

MIT
