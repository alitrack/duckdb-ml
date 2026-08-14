"""Kernel family: svm (linear/rbf) and svr (linear/rbf) — cross-checked against
scikit-learn SVC / SVR.

duckdb-ml kernel codes: 0=linear, 1=rbf (gaussian), 2=poly, 3=sigmoid.
The Rust SMO / working-set implementations differ from libsvm internals, so the
proof is predictive quality (accuracy / R² / prediction correlation).
"""

import numpy as np
from common import (
    Report, connect, train, predict, r2, accuracy, sign_corr, SEED,
)

rng = np.random.default_rng(SEED)
report = Report("verify_03_kernel")
con = connect()


def split(X, y, frac=0.7):
    k = int(len(y) * frac)
    return X[:k], X[k:], y[:k], y[k:]


# ── linearly separable blobs (svm linear) ──
Xl = np.vstack([rng.normal([-1.5, -1.5], 0.7, (150, 2)), rng.normal([1.5, 1.5], 0.7, (150, 2))])
yl = np.array([0.0] * 150 + [1.0] * 150)
Xlt, Xlv, ylt, ylv = split(Xl, yl)

# ── concentric circles (svm rbf) ──
n = 300
th = rng.uniform(0, 2 * np.pi, n)
Xc = np.column_stack([np.cos(th), np.sin(th)]) * (1.0 + 0.2 * rng.normal(size=(n, 2)))
Xc2 = np.column_stack([np.cos(th), np.sin(th)]) * (3.0 + 0.2 * rng.normal(size=(n, 2)))
Xr = np.vstack([Xc, Xc2])
yr = np.array([0.0] * n + [1.0] * n)
Xrt, Xrv, yrt, yrv = split(Xr, yr)

# ── svr linear: y = 3x + 1 (noise-free) ──
Xs = rng.uniform(-3, 3, 200)[:, None]
ys = 3.0 * Xs[:, 0] + 1.0
Xst, Xsv, yst, ysv = split(Xs, ys)

# ── svr rbf: y = sin(2x) ──
Xf = rng.uniform(-3, 3, 250)[:, None]
yf = np.sin(2.0 * Xf[:, 0])
Xft, Xfv, yft, yfv = split(Xf, yf)

from sklearn.svm import SVC, SVR

# ── 1. svm linear ──
train(con, "svm_lin", "svm", Xlt, ylt, {"c": 1.0, "kernel": 0})
p_ours = np.asarray(predict(con, "svm_lin", Xlv), float).round()
p_sk = SVC(kernel="linear", C=1.0, random_state=SEED).fit(Xlt, ylt).predict(Xlv)
acc_ours, acc_sk = accuracy(ylv, p_ours), accuracy(ylv, p_sk)
report.add("svm-linear accuracy(ours)", acc_ours, 0.95, 0.0, acc_ours >= 0.95,
           f"sklearn acc={acc_sk:.4f}")

# ── 2. svm rbf (circles) ──
train(con, "svm_rbf", "svm", Xrt, yrt, {"c": 1.0, "kernel": 1, "gamma": 0.5})
p_ours = np.asarray(predict(con, "svm_rbf", Xrv), float).round()
p_sk = SVC(kernel="rbf", C=1.0, gamma=0.5, random_state=SEED).fit(Xrt, yrt).predict(Xrv)
acc_ours, acc_sk = accuracy(yrv, p_ours), accuracy(yrv, p_sk)
report.add("svm-rbf accuracy(ours)", acc_ours, 0.95, 0.0, acc_ours >= 0.95,
           f"sklearn acc={acc_sk:.4f}")

# ── 3. svr linear: y = 3x + 1 ──
train(con, "svr_lin", "svr", Xst, yst, {"c": 10.0, "epsilon": 0.05, "kernel": 0, "tol": 1e-4})
p_ours = np.asarray(predict(con, "svr_lin", Xsv), float)
p_sk = SVR(kernel="linear", C=10.0, epsilon=0.05).fit(Xst, yst).predict(Xsv)
r2_ours, r2_sk = r2(ysv, p_ours), r2(ysv, p_sk)
report.add("svr-linear R2(ours)", r2_ours, 0.99, 0.0, r2_ours >= 0.99,
           f"sklearn R2={r2_sk:.6f}")
report.add("svr-linear pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 0.001,
           sign_corr(p_ours, p_sk) >= 0.999)

# ── 4. svr rbf: sin(x) + x² — fixed working-set SMO (2026-08-14) ──
# Previously unstable (R² 0.98 ↔ -0.12 across draws, x² ≈ -0.5) due to
# pair-sweep working set that pushed β to box boundaries on monotone targets.
# Root cause fixed: libsvm-style directional-gradient working set + unbiased
# gradients (bias held 0 in iteration, final bias from free SVs) + ε·sign(β)
# term in the pair update + incremental gradient maintenance.
Xs2 = rng.uniform(-2, 2, 200)[:, None]
ys2 = np.sin(Xs2[:, 0])
Xg, Xh, yg, yh = split(Xs2, ys2)
train(con, "svr_rbf", "svr", Xg, yg, {"c": 50.0, "epsilon": 1e-4, "kernel": 1, "gamma": 1.0, "tol": 1e-3, "max_iter": 20000})
p_ours = np.asarray(predict(con, "svr_rbf", Xh), float)
p_sk = SVR(kernel="rbf", C=50.0, epsilon=1e-4, gamma=1.0).fit(Xg, yg).predict(Xh)
r2_ours, r2_sk = r2(yh, p_ours), r2(yh, p_sk)
report.add("svr-rbf sin(x) R2(ours)", r2_ours, 0.9, 0.0, r2_ours >= 0.9,
           f"sklearn R2={r2_sk:.4f}")
report.add("svr-rbf sin(x) pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 0.01,
           sign_corr(p_ours, p_sk) >= 0.99)

# x² on [0,3] — the monotone target that exposed the old working-set bug
Xm = rng.uniform(0, 3, 200)[:, None]
ym = Xm[:, 0] ** 2
Xp, Xq, yp, yq = split(Xm, ym)
train(con, "svr_rbf2", "svr", Xp, yp, {"c": 50.0, "epsilon": 1e-4, "kernel": 1, "gamma": 1.0, "tol": 1e-3, "max_iter": 20000})
p2_ours = np.asarray(predict(con, "svr_rbf2", Xq), float)
p2_sk = SVR(kernel="rbf", C=50.0, epsilon=1e-4, gamma=1.0).fit(Xp, yp).predict(Xq)
r2b_ours, r2b_sk = r2(yq, p2_ours), r2(yq, p2_sk)
report.add("svr-rbf x^2 R2(ours)", r2b_ours, 0.9, 0.0, r2b_ours >= 0.9,
           f"sklearn R2={r2b_sk:.4f}")
report.add("svr-rbf x^2 pred corr vs sklearn", sign_corr(p2_ours, p2_sk), 1.0, 0.01,
           sign_corr(p2_ours, p2_sk) >= 0.99)

# ── 5. lda: supervised dim-reduction vs sklearn LinearDiscriminantAnalysis ──
# Generalized eigenproblem S_b v = λ S_w v solved via ridge-Cholesky +
# symmetrized M = L⁻¹S_bL⁻ᵀ + deflated power iteration (v = L⁻ᵀu back-transform).
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA_CENTERS = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
# interleaved by class so the sequential split keeps all 3 classes in both halves
Xl, yl = [], []
for _ in range(100):
    for lc, lcen in enumerate(LDA_CENTERS):
        Xl.append(rng.normal(lcen, 0.8))
        yl.append(float(lc))
Xl = np.array(Xl)
yl = np.array(yl)
Xlt, Xl2, ylt, yl2 = split(Xl, yl)
train(con, "lda_c", "lda", Xlt, ylt, {"k": 2})
pl_ours = np.asarray(predict(con, "lda_c", Xl2), float)
pl_sk = LinearDiscriminantAnalysis(n_components=2).fit(Xlt, ylt).transform(Xl2)[:, 0]
# direction sign is arbitrary → |corr|; sklearn scales are whitened → corr only
lcorr = abs(sign_corr(pl_ours, pl_sk))
report.add("lda first-discriminant corr vs sklearn", lcorr, 1.0, 0.001,
           lcorr >= 0.99, "sign-agnostic |corr| on held-out transform")
# class separation sanity: discriminant must separate at least one class pair
lmean = [pl_ours[yl2 == c].mean() for c in range(3)]
sep_ok = max(lmean) - min(lmean) > 1.0
report.add("lda class-separation (span>1)", max(lmean) - min(lmean), 2.0, 0.0,
           sep_ok, f"class means {[round(v, 3) for v in lmean]}")

ok = report.print_report()
con.close()
raise SystemExit(0 if ok else 1)
