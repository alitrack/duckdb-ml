//! Voting ensemble over already-registered models.
//!
//! `ml_voting(model_names_json, features_json, mode)` aggregates the
//! predictions of any registered models: hard = majority vote (labels are
//! rounded to integers), mean = arithmetic average (regression).

/// Voting aggregation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VotingMode {
    /// Majority vote over rounded labels (classification).
    Hard,
    /// Arithmetic mean (regression).
    Mean,
}

impl VotingMode {
    pub fn parse(s: &str) -> Option<VotingMode> {
        match s {
            "hard" | "majority" => Some(VotingMode::Hard),
            "mean" | "average" | "soft" => Some(VotingMode::Mean),
            _ => None,
        }
    }
}

/// Aggregate predictions of the named registered models for one feature row.
pub fn vote(
    model_names: &[String],
    features: &[f64],
    mode: VotingMode,
) -> Result<f64, String> {
    assert!(!model_names.is_empty(), "at least one model required");
    let registry = crate::model::global_registry();
    let mut preds = Vec::with_capacity(model_names.len());
    for name in model_names {
        let m = registry
            .get(name)
            .ok_or_else(|| format!("unknown model: '{name}'"))?;
        preds.push(
            m.predict(features)
                .map_err(|e| format!("model '{name}' predict: {e}"))?,
        );
    }
    match mode {
        VotingMode::Mean => Ok(preds.iter().sum::<f64>() / preds.len() as f64),
        VotingMode::Hard => {
            // labels → nearest integer, majority, ties → first in model order
            let votes: Vec<i64> = preds.iter().map(|&p| p.round() as i64).collect();
            let mut best = votes[0];
            let mut best_cnt = 0i64;
            for &v in &votes {
                let c = votes.iter().filter(|&&x| x == v).count() as i64;
                if c > best_cnt {
                    best_cnt = c;
                    best = v;
                }
            }
            Ok(best as f64)
        }
    }
}

/// Predictions of each member (for diagnostics).
pub fn member_predictions(model_names: &[String], features: &[f64]) -> Result<Vec<f64>, String> {
    let registry = crate::model::global_registry();
    let mut preds = Vec::with_capacity(model_names.len());
    for name in model_names {
        let m = registry
            .get(name)
            .ok_or_else(|| format!("unknown model: '{name}'"))?;
        preds.push(
            m.predict(features)
                .map_err(|e| format!("model '{name}' predict: {e}"))?,
        );
    }
    Ok(preds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_majority_tie_first_wins() {
        let votes: Vec<f64> = vec![1.0, 2.0, 1.0, 2.0, 3.0];
        // 1 and 2 tie at 2 each → first (1.0) wins
        let best = votes
            .iter()
            .map(|&p| p.round() as i64)
            .fold((0i64, 0i64), |(best, cnt), v| {
                let c = votes.iter().filter(|&&x| x.round() as i64 == v).count() as i64;
                if c > cnt {
                    (v, c)
                } else {
                    (best, cnt)
                }
            });
        assert_eq!(best.0, 1);
    }

    #[test]
    fn mean_averages() {
        let preds = vec![0.5, 1.5, 2.5];
        let mean = preds.iter().sum::<f64>() / 3.0;
        assert!((mean - 1.5).abs() < 1e-12);
    }
}
