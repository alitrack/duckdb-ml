use super::MlModel;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Tracks when a model was last used for LRU eviction
struct CacheEntry {
    model: Arc<dyn MlModel>,
    last_used: Instant,
}

/// Thread-safe model registry with LRU caching
pub struct ModelRegistry {
    models: RwLock<HashMap<String, CacheEntry>>,
    max_cached: usize,
}

impl ModelRegistry {
    pub fn new(max_cached: usize) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
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

        // Evict LRU if at capacity
        if models.len() >= self.max_cached && !models.contains_key(&name) {
            let oldest = models
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest {
                models.remove(&key);
                log::debug!("LRU evicted model: {}", key);
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
}

/// Global registry singleton
static REGISTRY: std::sync::LazyLock<ModelRegistry> =
    std::sync::LazyLock::new(|| ModelRegistry::new(10));

pub fn global_registry() -> &'static ModelRegistry {
    &REGISTRY
}
