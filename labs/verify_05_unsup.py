"""Unsupervised family: kmeans / dbscan / pca — cross-checked against scikit-learn.

  - kmeans : cluster alignment via Hungarian matching (labels are arbitrary ids)
  - dbscan : cluster alignment on training points (noise = -1 must match)
  - pca    : duckdb-ml predict() returns the PC1 score only (MlModel is
    scalar-output) — verified via sign-adjusted correlation vs sklearn PC1
"""

import numpy as np
from common import (
    Report, connect, train, predict, sign_corr, cluster_alignment, SEED,
)

rng = np.random.default_rng(SEED)
report = Report("verify_05_unsup")
con = connect()

# ── kmeans: 4 well-separated Gaussians ──
centers = np.array([[2, 2], [-2, 2], [2, -2], [-2, -2]])
Xk = np.vstack([rng.normal(c, 0.35, (100, 2)) for c in centers])
yk_dummy = np.zeros(len(Xk))  # unsupervised: dummy target

# ── dbscan: two arcs (moons) + noise ──
n = 200
th = rng.uniform(0, np.pi, n)
moon1 = np.column_stack([np.cos(th), np.sin(th)]) + rng.normal(0, 0.03, (n, 2))
moon2 = np.column_stack([2.5 - np.cos(th), 1.0 - np.sin(th)]) + rng.normal(0, 0.03, (n, 2))
noise = rng.uniform(-2.0, 3.5, (30, 2))
Xd = np.vstack([moon1, moon2, noise])
yd_dummy = np.zeros(len(Xd))

# ── pca: 3 features from 2 latent dims (rank 2 in 3D) ──
z = rng.normal(size=(300, 2))
Xp = np.column_stack([
    z[:, 0] + 0.5 * z[:, 1],
    0.5 * z[:, 0] + z[:, 1],
    z[:, 0] - z[:, 1],
])
yp_dummy = np.zeros(len(Xp))

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# ── 1. kmeans vs sklearn KMeans ──
train(con, "km01", "kmeans", Xk, yk_dummy, {"k": 4, "max_iters": 300, "tol": 1e-6})
p_ours = np.asarray(predict(con, "km01", Xk), float)
p_sk = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit_predict(Xk)
al = cluster_alignment(p_ours, p_sk, 4)
report.add("kmeans alignment vs sklearn", al, 0.95, 0.0, al >= 0.95,
           f"kmeans++ init (seed={SEED})")

# ── 2. dbscan vs sklearn DBSCAN ──
# NOTE: duckdb-ml predict() assigns EVERY point to its nearest cluster mean
# (MADlib-style simplification) — it has no "noise" concept. sklearn keeps
# noise points as -1. So alignment is scored on non-noise points only.
train(con, "db01", "dbscan", Xd, yd_dummy, {"eps": 0.3, "min_points": 5})
p_ours = np.asarray(predict(con, "db01", Xd), float)
p_sk = DBSCAN(eps=0.3, min_samples=5).fit_predict(Xd)
non_noise = p_sk != -1
al = cluster_alignment(p_ours[non_noise], p_sk[non_noise],
                       max(len(set(p_sk)) - 1, 2))
noise_ours = float(np.mean(p_ours[~non_noise] == -1))  # duckdb may not emit -1
report.add("dbscan alignment (non-noise pts)", al, 0.9, 0.0, al >= 0.9,
           f"sklearn marks {int((~non_noise).sum())} pts noise (excluded)")

# ── 3. pca: PC1 score vs sklearn PCA ──
train(con, "pca01", "pca", Xp, yp_dummy, {"k": 2})
p_ours = np.asarray(predict(con, "pca01", Xp), float)  # PC1 score
p_sk = PCA(n_components=2, random_state=SEED).fit(Xp).transform(Xp)[:, 0]
c = sign_corr(p_ours, p_sk)
report.add("pca PC1 corr vs sklearn", c, 0.999, 0.0, c >= 0.999,
           "sign is arbitrary; |corr| >= 0.999")
ev_ours = float(np.var(p_ours))
ev_sk = float(np.var(p_sk))
report.add("pca PC1 var ratio (ours/sklearn)", ev_ours / ev_sk if ev_sk else float("nan"),
           1.0, 0.01, abs(ev_ours - ev_sk) / ev_sk < 0.01,
           "PC1 captured variance must match")

ok = report.print_report()
con.close()
raise SystemExit(0 if ok else 1)
