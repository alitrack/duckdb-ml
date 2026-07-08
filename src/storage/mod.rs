use duckdb::Connection;
use std::error::Error;

/// Create/upgrade duckdb_ml management tables.
/// Idempotent — skips if table/column already exists.
pub fn ensure_tables(con: &Connection) -> Result<(), Box<dyn Error>> {
    // ── v0.9 compat: original models table (kept for backward compat) ──
    con.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS duckdb_ml.models (
            name TEXT PRIMARY KEY,
            algorithm TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'ready',
            metadata JSON,
            r_squared DOUBLE,
            mse DOUBLE
        );
        ",
    )?;

    // ── v0.10: versioned models table ──
    con.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS duckdb_ml.models_v2 (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            algorithm TEXT NOT NULL,
            runtime TEXT NOT NULL DEFAULT 'rust',
            hyperparams JSON,
            metrics JSON,
            status TEXT DEFAULT 'ready',
            snapshot_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, version)
        );
        ",
    )?;

    // ── v0.10: data snapshots ──
    con.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS duckdb_ml.snapshots (
            id INTEGER PRIMARY KEY,
            relation_name TEXT NOT NULL,
            n_features INTEGER NOT NULL,
            n_samples INTEGER NOT NULL,
            target_column TEXT,
            feature_columns JSONB,
            data_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ",
    )?;

    // ── v0.10: deployments — which model version is live ──
    con.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS duckdb_ml.deployments (
            id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES duckdb_ml.models_v2(id)
        );
        CREATE INDEX IF NOT EXISTS idx_deployments_model_name
            ON duckdb_ml.deployments(model_name, created_at DESC);
        ",
    )?;

    // ── v0.10: experiment tracking ──
    con.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS duckdb_ml.experiments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            task TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS duckdb_ml.runs (
            id INTEGER PRIMARY KEY,
            experiment_id INTEGER NOT NULL,
            run_name TEXT NOT NULL,
            model_id INTEGER,
            status TEXT DEFAULT 'running',
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES duckdb_ml.experiments(id),
            FOREIGN KEY (model_id) REFERENCES duckdb_ml.models_v2(id)
        );

        CREATE TABLE IF NOT EXISTS duckdb_ml.metrics (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value DOUBLE NOT NULL,
            step INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES duckdb_ml.runs(id)
        );

        CREATE TABLE IF NOT EXISTS duckdb_ml.params (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES duckdb_ml.runs(id)
        );
        ",
    )?;

    Ok(())
}

/// Save a versioned model to duckdb_ml.models_v2
pub fn save_model_v2(
    con: &mut Connection,
    name: &str,
    algorithm: &str,
    hyperparams: &str,
    metrics_json: &str,
    snapshot_id: Option<i64>,
) -> Result<i64, Box<dyn Error>> {
    let sn = snapshot_id
        .map(|v| v.to_string())
        .unwrap_or_else(|| "NULL".into());

    let sql = format!(
        "INSERT INTO duckdb_ml.models_v2 (name, version, algorithm, hyperparams, metrics, snapshot_id)
         VALUES (
             '{name}',
             COALESCE((SELECT MAX(version) + 1 FROM duckdb_ml.models_v2 WHERE name = '{name}'), 1),
             '{algo}',
             '{hp}',
             '{m}',
             {sn}
         )
         RETURNING id",
        name = name.replace('\'', "''"),
        algo = algorithm,
        hp = hyperparams.replace('\'', "''"),
        m = metrics_json.replace('\'', "''"),
    );

    let mut stmt = con.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(row.get(0)?)
    } else {
        Err("Failed to insert model".into())
    }
}

/// Deploy a model with a strategy
pub fn deploy_model(
    con: &mut Connection,
    model_name: &str,
    strategy: &str,
) -> Result<i64, Box<dyn Error>> {
    let model_id_sql = match strategy {
        "best_score" => {
            format!(
                "SELECT id FROM duckdb_ml.models_v2
                 WHERE name = '{n}'
                 ORDER BY COALESCE((metrics->>'r_squared')::DOUBLE, (metrics->>'accuracy')::DOUBLE, 0) DESC
                 LIMIT 1",
                n = model_name.replace('\'', "''"),
            )
        }
        "most_recent" => {
            format!(
                "SELECT id FROM duckdb_ml.models_v2 WHERE name = '{n}' ORDER BY created_at DESC LIMIT 1",
                n = model_name.replace('\'', "''"),
            )
        }
        "rollback" => {
            format!(
                "SELECT m.id FROM duckdb_ml.models_v2 m
                 JOIN duckdb_ml.deployments d2 ON d2.model_id = (
                     SELECT d.model_id FROM duckdb_ml.deployments d
                     WHERE d.model_name = '{n}'
                     ORDER BY d.created_at DESC LIMIT 1 OFFSET 1
                 )
                 WHERE m.name = '{n}'
                 LIMIT 1",
                n = model_name.replace('\'', "''"),
            )
        }
        _ => {
            return Err(format!("Unknown deploy strategy: {strategy}").into());
        }
    };

    let mut stmt = con.prepare(&model_id_sql)?;
    let mut rows = stmt.query([])?;
    let model_id: i64 = if let Some(row) = rows.next()? {
        row.get(0)?
    } else {
        return Err(format!("No model found for '{model_name}' with strategy '{strategy}'").into());
    };

    let insert_sql = format!(
        "INSERT INTO duckdb_ml.deployments (model_name, model_id, strategy) VALUES ('{n}', {id}, '{s}') RETURNING id",
        n = model_name.replace('\'', "''"),
        id = model_id,
        s = strategy,
    );

    let mut stmt = con.prepare(&insert_sql)?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(row.get(0)?)
    } else {
        Err("Failed to insert deployment".into())
    }
}

/// Get the currently deployed model_id for a model name
pub fn get_deployed_model_id(
    con: &Connection,
    model_name: &str,
) -> Result<Option<i64>, Box<dyn Error>> {
    let sql = format!(
        "SELECT model_id FROM duckdb_ml.deployments
         WHERE model_name = '{n}'
         ORDER BY created_at DESC LIMIT 1",
        n = model_name.replace('\'', "''"),
    );
    let mut stmt = con.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(Some(row.get(0)?))
    } else {
        Ok(None)
    }
}

/// Save a snapshot of training data
pub fn save_snapshot(
    con: &mut Connection,
    relation_name: &str,
    n_features: usize,
    n_samples: usize,
    target_column: Option<&str>,
    feature_columns: &[String],
    data_hash: Option<&str>,
) -> Result<i64, Box<dyn Error>> {
    let fc_json = serde_json::to_string(feature_columns).unwrap_or_else(|_| "[]".into());
    let tc = target_column
        .map(|v| format!("'{}'", v.replace('\'', "''")))
        .unwrap_or_else(|| "NULL".into());
    let dh = data_hash
        .map(|v| format!("'{}'", v.replace('\'', "''")))
        .unwrap_or_else(|| "NULL".into());

    let sql = format!(
        "INSERT INTO duckdb_ml.snapshots (relation_name, n_features, n_samples, target_column, feature_columns, data_hash)
         VALUES ('{rn}', {nf}, {ns}, {tc}, '{fc}', {dh})
         RETURNING id",
        rn = relation_name.replace('\'', "''"),
        nf = n_features,
        ns = n_samples,
        tc = tc,
        fc = fc_json.replace('\'', "''"),
        dh = dh,
    );

    let mut stmt = con.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(row.get(0)?)
    } else {
        Err("Failed to insert snapshot".into())
    }
}

// ── Experiment tracking ──

/// Start an experiment (idempotent)
pub fn ensure_experiment(
    con: &mut Connection,
    name: &str,
    task: Option<&str>,
) -> Result<i64, Box<dyn Error>> {
    let task_str = task
        .map(|t| format!("'{}'", t.replace('\'', "''")))
        .unwrap_or_else(|| "NULL".into());
    let sql = format!(
        "INSERT INTO duckdb_ml.experiments (name, task)
         VALUES ('{n}', {t})
         ON CONFLICT (name) DO UPDATE SET task = COALESCE(duckdb_ml.experiments.task, {t})
         RETURNING id",
        n = name.replace('\'', "''"),
        t = task_str,
    );
    let mut stmt = con.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(row.get(0)?)
    } else {
        Err("Failed to ensure experiment".into())
    }
}

/// Start a run
pub fn start_run(
    con: &mut Connection,
    experiment_id: i64,
    run_name: &str,
) -> Result<i64, Box<dyn Error>> {
    let sql = format!(
        "INSERT INTO duckdb_ml.runs (experiment_id, run_name) VALUES ({eid}, '{rn}') RETURNING id",
        eid = experiment_id,
        rn = run_name.replace('\'', "''"),
    );
    let mut stmt = con.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(row.get(0)?)
    } else {
        Err("Failed to start run".into())
    }
}

/// Finish a run
pub fn finish_run(
    con: &mut Connection,
    run_id: i64,
    model_id: Option<i64>,
) -> Result<(), Box<dyn Error>> {
    let mid = model_id
        .map(|v| v.to_string())
        .unwrap_or_else(|| "NULL".into());
    let sql = format!(
        "UPDATE duckdb_ml.runs SET status = 'finished', end_time = CURRENT_TIMESTAMP, model_id = {mid}
         WHERE id = {rid}",
        rid = run_id,
        mid = mid,
    );
    con.execute(&sql, [])?;
    Ok(())
}

/// Log a metric
pub fn log_metric(
    con: &mut Connection,
    run_id: i64,
    key: &str,
    value: f64,
    step: i64,
) -> Result<(), Box<dyn Error>> {
    let sql = format!(
        "INSERT INTO duckdb_ml.metrics (run_id, key, value, step) VALUES ({rid}, '{k}', {v}, {st})",
        rid = run_id,
        k = key.replace('\'', "''"),
        v = value,
        st = step,
    );
    con.execute(&sql, [])?;
    Ok(())
}

/// Log a parameter
pub fn log_param(
    con: &mut Connection,
    run_id: i64,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn Error>> {
    let sql = format!(
        "INSERT INTO duckdb_ml.params (run_id, key, value) VALUES ({rid}, '{k}', '{v}')",
        rid = run_id,
        k = key.replace('\'', "''"),
        v = value.replace('\'', "''"),
    );
    con.execute(&sql, [])?;
    Ok(())
}

// ── Legacy helpers (kept for backward compat) ──

/// Load model weights from DuckDB tables (v0.9 compat)
pub fn load_weights(con: &Connection, name: &str) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    let sql = format!(
        "SELECT weights FROM duckdb_ml.model_weights WHERE model_name = '{}'",
        name.replace('\'', "''"),
    );
    let mut stmt = con.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    match rows.next()? {
        Some(row) => {
            let blob: Vec<u8> = row.get(0)?;
            Ok(Some(blob))
        }
        None => Ok(None),
    }
}

/// Delete a model from both tables (v0.9 compat)
pub fn delete_model(con: &Connection, name: &str) -> Result<(), Box<dyn Error>> {
    let escaped = name.replace('\'', "''");
    con.execute(
        &format!(
            "DELETE FROM duckdb_ml.model_weights WHERE model_name = '{}'",
            escaped
        ),
        [],
    )?;
    con.execute(
        &format!("DELETE FROM duckdb_ml.models WHERE name = '{}'", escaped),
        [],
    )?;
    Ok(())
}

/// List all models (v0.9 compat)
pub fn list_models(con: &Connection) -> Result<Vec<ModelInfo>, Box<dyn Error>> {
    let mut stmt = con.prepare(
        "SELECT name, algorithm, created_at, status, r_squared, mse FROM duckdb_ml.models ORDER BY created_at DESC",
    )?;
    let mut rows = stmt.query([])?;
    let mut models = Vec::new();
    while let Some(row) = rows.next()? {
        models.push(ModelInfo {
            name: row.get(0)?,
            algorithm: row.get(1)?,
            created_at: row.get::<_, String>(2).unwrap_or_default(),
            status: row.get::<_, String>(3).unwrap_or_default(),
            r_squared: row.get(4)?,
            mse: row.get(5)?,
        });
    }
    Ok(models)
}

// ── v0.10: List versioned models ──
pub fn list_models_v2(con: &Connection) -> Result<Vec<ModelInfoV2>, Box<dyn Error>> {
    let mut stmt = con.prepare(
        "SELECT m.id, m.name, m.version, m.algorithm, m.runtime, m.status, m.created_at,
                COALESCE(m.metrics->>'r_squared', m.metrics->>'accuracy') as score,
                d.model_id IS NOT NULL as deployed
         FROM duckdb_ml.models_v2 m
         LEFT JOIN (
             SELECT DISTINCT ON(model_name) model_name, model_id
             FROM duckdb_ml.deployments
             ORDER BY model_name, created_at DESC
         ) d ON d.model_id = m.id
         ORDER BY m.created_at DESC",
    )?;
    let mut rows = stmt.query([])?;
    let mut models = Vec::new();
    while let Some(row) = rows.next()? {
        models.push(ModelInfoV2 {
            id: row.get(0)?,
            name: row.get(1)?,
            version: row.get(2)?,
            algorithm: row.get(3)?,
            runtime: row.get(4)?,
            status: row.get(5)?,
            created_at: row.get::<_, String>(6).unwrap_or_default(),
            score: row.get(7)?,
            deployed: row.get(8)?,
        });
    }
    Ok(models)
}

#[derive(Debug)]
pub struct ModelInfo {
    pub name: String,
    pub algorithm: String,
    pub created_at: String,
    pub status: String,
    pub r_squared: Option<f64>,
    pub mse: Option<f64>,
}

#[derive(Debug)]
pub struct ModelInfoV2 {
    pub id: i64,
    pub name: String,
    pub version: i64,
    pub algorithm: String,
    pub runtime: String,
    pub status: String,
    pub created_at: String,
    pub score: Option<f64>,
    pub deployed: bool,
}
