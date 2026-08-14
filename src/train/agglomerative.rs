//! Agglomerative hierarchical clustering (single / complete / average
//! linkage). Greedy bottom-up merging of the nearest clusters until `k`
//! remain; deterministic (no randomness). Labels are renumbered 0..k in the
//! order clusters first appear.

/// Linkage criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    Single,
    Complete,
    Average,
}

impl Linkage {
    pub fn parse(s: &str) -> Option<Linkage> {
        match s {
            "single" => Some(Linkage::Single),
            "complete" => Some(Linkage::Complete),
            "average" => Some(Linkage::Average),
            _ => None,
        }
    }
}

/// Agglomerative result.
pub struct AggResult {
    /// Cluster label per sample (0..k, ordered by first appearance).
    pub labels: Vec<usize>,
    /// Cluster centroids.
    pub centers: Vec<Vec<f64>>,
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Train agglomerative clustering.
pub fn train(x: &[Vec<f64>], n_clusters: usize, linkage: Linkage) -> AggResult {
    assert!(!x.is_empty());
    let n = x.len();
    let k = n_clusters.min(n).max(1);
    if n == 1 || k == n {
        let labels: Vec<usize> = (0..n).collect();
        let centers = x.to_vec();
        return AggResult { labels, centers };
    }

    // each sample starts as its own cluster
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active = n;

    // pairwise cluster distance: maintain a heap-free O(n²) matrix is fine
    // for moderate n; update lazily by recomputing from members.
    while active > k {
        // find the closest pair of clusters
        let mut best = (f64::INFINITY, 0usize, 1usize);
        for i in 0..n {
            if members[i].is_empty() {
                continue;
            }
            for j in (i + 1)..n {
                if members[j].is_empty() {
                    continue;
                }
                let d = cluster_dist(&members[i], &members[j], x, linkage);
                if d < best.0 {
                    best = (d, i, j);
                }
            }
        }
        let (_, i, j) = best;
        let mut merged = members[i].clone();
        merged.extend(members[j].iter().copied());
        members[i] = merged;
        members[j].clear();
        active -= 1;
    }

    // renumber 0..k by first-appearance order of the surviving clusters
    let mut labels = vec![0usize; n];
    let mut label_of = vec![0usize; n]; // cluster idx → label
    let mut next = 0usize;
    let mut centers = Vec::with_capacity(k);
    for ci in 0..n {
        if members[ci].is_empty() {
            continue;
        }
        label_of[ci] = next;
        let m = &members[ci];
        let mut c = vec![0.0f64; x[0].len()];
        for &si in m {
            for (d, v) in x[si].iter().enumerate() {
                c[d] += v;
            }
        }
        for v in c.iter_mut() {
            *v /= m.len() as f64;
        }
        centers.push(c);
        next += 1;
    }
    for (si, lab) in labels.iter_mut().enumerate() {
        let ci = (0..n).find(|&ci| members[ci].contains(&si)).unwrap();
        *lab = label_of[ci];
    }

    AggResult { labels, centers }
}

fn cluster_dist(a: &[usize], b: &[usize], x: &[Vec<f64>], linkage: Linkage) -> f64 {
    match linkage {
        Linkage::Single => a
            .iter()
            .flat_map(|&i| b.iter().map(move |&j| dist2(&x[i], &x[j])))
            .fold(f64::INFINITY, f64::min),
        Linkage::Complete => a
            .iter()
            .flat_map(|&i| b.iter().map(move |&j| dist2(&x[i], &x[j])))
            .fold(0.0f64, f64::max),
        Linkage::Average => {
            let mut s = 0.0f64;
            let mut cnt = 0usize;
            for &i in a {
                for &j in b {
                    s += dist2(&x[i], &x[j]);
                    cnt += 1;
                }
            }
            s / cnt as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_blobs() -> Vec<Vec<f64>> {
        // two tight blobs at (0,0)/(10,0), one at (0,10)
        let mut x = Vec::new();
        for _ in 0..10 {
            x.push(vec![0.0, 0.0]);
        }
        for _ in 0..10 {
            x.push(vec![10.0, 0.0]);
        }
        for _ in 0..10 {
            x.push(vec![0.0, 10.0]);
        }
        x
    }

    #[test]
    fn finds_three_blobs() {
        let x = three_blobs();
        let r = train(&x, 3, Linkage::Complete);
        assert_eq!(r.labels.len(), 30);
        // each blob is internally consistent (same label within each of the
        // three 10-sample groups)
        for g in 0..3 {
            let l = r.labels[g * 10];
            for i in 1..10 {
                assert_eq!(r.labels[g * 10 + i], l, "blob {g} split");
            }
        }
        // three distinct labels
        let mut s: Vec<usize> = r.labels.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn linkage_variants_agree_on_separated_blobs() {
        let x = three_blobs();
        for lnk in [Linkage::Single, Linkage::Complete, Linkage::Average] {
            let r = train(&x, 3, lnk);
            let mut s: Vec<usize> = r.labels.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), 3, "{lnk:?}");
        }
    }

    #[test]
    fn k_equals_n_returns_singletons() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let r = train(&x, 3, Linkage::Single);
        assert_eq!(r.labels, vec![0, 1, 2]);
    }

    #[test]
    fn deterministic() {
        let x = three_blobs();
        let a = train(&x, 3, Linkage::Average);
        let b = train(&x, 3, Linkage::Average);
        assert_eq!(a.labels, b.labels);
    }
}
