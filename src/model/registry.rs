use super::MlModel;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Tracks when a model was last used for LRU eviction
struct CacheEntry {
    model: Arc<dyn MlModel>,
    last_used: Instant,
}

/// Deployment state for a model name
#[derive(Debug, Clone)]
pub struct Deployment {
    pub model_key: String,
    pub strategy: String,
    pub deployed_at: Instant,
}

/// Thread-safe model registry with LRU caching, deployment tracking, and data snapshots
pub struct ModelRegistry {
    models: RwLock<HashMap<String, CacheEntry>>,
    /// model_name -> current deployment
    deployments: RwLock<HashMap<String, Deployment>>,
    /// model_name -> previous deployments (for rollback)
    deployment_history: RwLock<HashMap<String, Vec<Deployment>>>,
    /// Data snapshots: (model_name, snapshot_name) -> snapshot metadata
    snapshots: RwLock<Vec<DataSnapshot>>,
    max_cached: usize,
}

/// Metadata about a training data snapshot
#[derive(Debug, Clone)]
pub struct DataSnapshot {
    pub model_name: String,
    pub relation_name: String,
    pub n_features: usize,
    pub n_samples: usize,
    pub target_column: String,
    pub feature_columns: Vec<String>,
    pub data_hash: String,
}

impl ModelRegistry {
    pub fn new(max_cached: usize) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            deployments: RwLock::new(HashMap::new()),
            deployment_history: RwLock::new(HashMap::new()),
            snapshots: RwLock::new(Vec::new()),
            max_cached,
        }
    }

    /// Get a model from cache or return None
    pub fn get(&self, name: &str) -> Option<Arc<dyn MlModel>> {
        let mut models = self.models.write().unwrap();
        if let Some(entry) = models.get_mut(name) {
            entry.last_used = Instant::now();
            Some(entry.model.clone())
        } else {
            None
        }
    }

    /// Insert a model into cache, evicting LRU if at capacity
    pub fn insert(&self, name: String, model: Arc<dyn MlModel>) {
        let mut models = self.models.write().unwrap();

        if models.len() >= self.max_cached && !models.contains_key(&name) {
            let oldest = models
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest {
                models.remove(&key);
            }
        }

        models.insert(
            name,
            CacheEntry {
                model,
                last_used: Instant::now(),
            },
        );
    }

    /// Remove a model from cache
    pub fn remove(&self, name: &str) {
        self.models.write().unwrap().remove(name);
    }

    /// List all cached model names
    pub fn list(&self) -> Vec<String> {
        self.models.read().unwrap().keys().cloned().collect()
    }

    /// Number of cached models
    pub fn len(&self) -> usize {
        self.models.read().unwrap().len()
    }

    /// Check if registry is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── Deployment management ──

    /// Deploy a model with a strategy.
    /// - "best_score": pick the model with the highest r_squared across all keys matching `name_*`
    /// - "most_recent": use the exact model_key
    /// - "rollback": restore the previous deployment
    /// - "specific": the model_key is the exact key
    pub fn deploy(
        &self,
        model_name: &str,
        strategy: &str,
        model_key: &str,
    ) -> Result<Deployment, String> {
        let actual_key = match strategy {
            "best_score" => {
                // Find the key with highest r_squared among models matching the prefix
                let models = self.models.read().unwrap();
                let prefix = format!("{model_name}_");
                let mut best_key = None;
                let mut best_r2 = f64::NEG_INFINITY;
                for key in models.keys() {
                    #[allow(clippy::collapsible_if)]
                    if key == model_name || key.starts_with(&prefix) {
                        if let Some(entry) = models.get(key) {
                            let r2 = entry
                                .model
                                .metadata()
                                .r_squared
                                .unwrap_or(f64::NEG_INFINITY);
                            if r2 > best_r2 {
                                best_r2 = r2;
                                best_key = Some(key.clone());
                            }
                        }
                    }
                }
                best_key
                    .ok_or_else(|| format!("No model found for '{model_name}' with best_score"))?
            }
            "most_recent" => model_key.to_string(),
            "rollback" => {
                let hist = self.deployment_history.read().unwrap();
                let history = hist
                    .get(model_name)
                    .ok_or("No deployment history for rollback")?;
                if history.len() < 2 {
                    return Err("Need at least 2 deployments to rollback".into());
                }
                history[history.len() - 2].model_key.clone()
            }
            _ => model_key.to_string(),
        };

        // Verify model exists
        if self.get(&actual_key).is_none() {
            return Err(format!("Model '{actual_key}' not found in registry"));
        }

        let deployment = Deployment {
            model_key: actual_key.clone(),
            strategy: strategy.to_string(),
            deployed_at: Instant::now(),
        };

        // Save current deployment to history before replacing
        {
            let mut deps = self.deployments.write().unwrap();
            if let Some(old) = deps.get(model_name) {
                self.deployment_history
                    .write()
                    .unwrap()
                    .entry(model_name.to_string())
                    .or_default()
                    .push(old.clone());
            }
            deps.insert(model_name.to_string(), deployment.clone());
        }

        Ok(deployment)
    }

    /// Get the currently deployed model key for a model name
    pub fn get_deployed(&self, model_name: &str) -> Option<String> {
        self.deployments
            .read()
            .unwrap()
            .get(model_name)
            .map(|d| d.model_key.clone())
    }

    /// Get deployment info
    pub fn get_deployment_info(&self) -> Vec<(String, String, String)> {
        self.deployments
            .read()
            .unwrap()
            .iter()
            .map(|(name, dep)| {
                let algo = self
                    .models
                    .read()
                    .unwrap()
                    .get(&dep.model_key)
                    .map(|e| e.model.algorithm().to_string())
                    .unwrap_or_default();
                (name.clone(), dep.model_key.clone(), algo)
            })
            .collect()
    }

    /// Get the model for the currently deployed version of a name.
    /// Falls back to exact name lookup if no deployment exists.
    pub fn get_deployed_model(&self, model_name: &str) -> Option<Arc<dyn MlModel>> {
        // Try deployment first
        #[allow(clippy::collapsible_if)]
        if let Some(key) = self.get_deployed(model_name) {
            if let Some(m) = self.get(&key) {
                return Some(m);
            }
        }
        // Fall back to exact name
        self.get(model_name)
    }

    // ── Data snapshot tracking ──

    /// Register a data snapshot
    pub fn add_snapshot(&self, snap: DataSnapshot) {
        self.snapshots.write().unwrap().push(snap);
    }

    /// List snapshots for a model name
    pub fn list_snapshots(&self, model_name: &str) -> Vec<DataSnapshot> {
        self.snapshots
            .read()
            .unwrap()
            .iter()
            .filter(|s| s.model_name == model_name)
            .cloned()
            .collect()
    }
}

/// Global registry singleton (max 100 models)
static REGISTRY: std::sync::LazyLock<ModelRegistry> =
    std::sync::LazyLock::new(|| ModelRegistry::new(100));

pub fn global_registry() -> &'static ModelRegistry {
    &REGISTRY
}
