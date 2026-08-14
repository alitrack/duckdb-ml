//! Kaplan-Meier non-parametric survival estimator.
//!
//! Classic product-limit estimate: sort unique event times t_j, risk set
//! n_j = #{t_i >= t_j}, events d_j at t_j, S(t_j) = S(t_{j-1})·(1 − d_j/n_j).
//! Censored observations (event=0) stay in the risk set until their time.

/// Kaplan-Meier result: survival staircase.
#[derive(Clone, Debug)]
pub struct KmResult {
    /// Unique event times (ascending).
    pub times: Vec<f64>,
    /// Survival probability at each event time (right-continuous).
    pub survival: Vec<f64>,
    /// Total observations.
    pub n: usize,
    /// Number of events.
    pub events: usize,
}

/// Product-limit estimate from (time, event) pairs.
pub fn train(time: &[f64], event: &[f64]) -> KmResult {
    assert_eq!(time.len(), event.len(), "time/event length mismatch");
    assert!(!time.is_empty(), "empty survival data");

    let n = time.len();
    // unique event times ascending
    let mut times: Vec<f64> = time
        .iter()
        .zip(event.iter())
        .filter(|(_, &e)| e > 0.0)
        .map(|(t, _)| *t)
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times.dedup();

    let mut survival = Vec::with_capacity(times.len());
    let mut s = 1.0f64;
    let mut events_total = 0usize;
    for &t in &times {
        let n_at_risk: usize = time.iter().filter(|&&ti| ti >= t).count();
        let d: usize = time
            .iter()
            .zip(event.iter())
            .filter(|(ti, e)| **ti == t && **e > 0.0)
            .count();
        events_total += d;
        if n_at_risk > 0 {
            s *= 1.0 - d as f64 / n_at_risk as f64;
        }
        survival.push(s);
    }

    KmResult {
        times,
        survival,
        n,
        events: events_total,
    }
}

/// Median survival time: first t with S(t) <= 0.5 (NaN if never reached).
pub fn median_survival(r: &KmResult) -> Option<f64> {
    for (i, &t) in r.times.iter().enumerate() {
        if r.survival[i] <= 0.5 {
            return Some(t);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_censoring_product_limit() {
        // 3 patients, all die at t=1,2,3
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1.0, 1.0, 1.0];
        let r = train(&time, &event);
        assert_eq!(r.times, vec![1.0, 2.0, 3.0]);
        // S(1)=2/3, S(2)=1/3, S(3)=0
        assert!((r.survival[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((r.survival[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!(r.survival[2].abs() < 1e-12);
        assert_eq!(median_survival(&r), Some(2.0));
    }

    #[test]
    fn censoring_removes_from_risk_set_after_its_time() {
        // t=1 dies, t=2 censored (leaves the risk set at t=2), t=3 dies
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1.0, 0.0, 1.0];
        let r = train(&time, &event);
        // t=1: risk=3, d=1 → 2/3. t=3: risk = {t=3} only (t=2 censored left),
        // d=1 → S(3) = (2/3)·(1−1/1) = 0
        assert_eq!(r.times, vec![1.0, 3.0]);
        assert!((r.survival[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!(r.survival[1].abs() < 1e-12);
    }

    #[test]
    fn median_absent_when_curve_stays_above_half() {
        // 1 event among 5 → S(1)=0.8, never crosses 0.5
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let r = train(&time, &event);
        assert!((r.survival[0] - 0.8).abs() < 1e-12);
        assert!(median_survival(&r).is_none());
    }

    #[test]
    fn all_censored_no_event_times() {
        let time = vec![1.0, 2.0, 5.0];
        let event = vec![0.0, 0.0, 0.0];
        let r = train(&time, &event);
        assert!(r.times.is_empty());
        assert_eq!(r.events, 0);
    }
}
