use duckdb::Connection;
use std::error::Error;

/// Ensure duckdb_ml management tables exist
pub fn ensure_tables(con: &Connection) -> Result<(), Box<dyn Error>> {
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

        CREATE TABLE IF NOT EXISTS duckdb_ml.model_weights (
            model_name TEXT PRIMARY KEY,
            weights BLOB NOT NULL,
            FOREIGN KEY (model_name) REFERENCES duckdb_ml.models(name)
        );
        ",
    )?;
    Ok(())
}

/// Save a model to DuckDB tables
pub fn save_model(
    con: &mut Connection,
    name: &str,
    algorithm: &str,
    metadata_json: &str,
    r_squared: Option<f64>,
    mse: Option<f64>,
    weights: &[u8],
) -> Result<(), Box<dyn Error>> {
    let rs = r_squared
        .map(|v| v.to_string())
        .unwrap_or_else(|| "NULL".into());
    let ms = mse.map(|v| v.to_string()).unwrap_or_else(|| "NULL".into());

    // DuckDB doesn't support parameterized BLOB well through the API
    // Use hex encoding for the weights blob
    let hex_weights: String = weights.iter().map(|b| format!("{:02x}", b)).collect();

    let sql = format!(
        "INSERT OR REPLACE INTO duckdb_ml.models (name, algorithm, metadata, r_squared, mse) \
         VALUES ('{}', '{}', '{}', {}, {})",
        name.replace('\'', "''"),
        algorithm,
        metadata_json.replace('\'', "''"),
        rs,
        ms,
    );

    con.execute(&sql, [])?;

    let weight_sql = format!(
        "INSERT OR REPLACE INTO duckdb_ml.model_weights (model_name, weights) \
         VALUES ('{}', from_hex('{}'))",
        name.replace('\'', "''"),
        hex_weights,
    );

    con.execute(&weight_sql, [])?;

    Ok(())
}

/// Load model weights from DuckDB tables
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

/// Delete a model from both tables
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

/// List all models
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

#[derive(Debug)]
pub struct ModelInfo {
    pub name: String,
    pub algorithm: String,
    pub created_at: String,
    pub status: String,
    pub r_squared: Option<f64>,
    pub mse: Option<f64>,
}
