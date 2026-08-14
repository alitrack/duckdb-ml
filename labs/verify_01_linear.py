"""Linear family: linear_regression / ridge_regression / lasso_regression /
elastic_net / robust + ml_ols — cross-checked against scikit-learn.

Evidence levels:
  - ml_ols            : coefficient-level match vs sklearn LinearRegression (1e-6)
  - linear_regression : prediction-level match (R² on held-out test)
  - ridge/lasso/elastic/robust: prediction correlation + R² vs sklearn twins
"""

import numpy as np
from common import (
    Report, connect, train, predict, ml_ols, r2, sign_corr, SEED,
)

rng = np.random.default_rng(SEED)
report = Report("verify_01_linear")
con = connect()


def split(X, y, frac=0.7):
    n = len(y)
    k = int(n * frac)
    return X[:k], X[k:], y[:k], y[k:]


# ── data: y = 2 + 3·x0 − 1.5·x1 + 0.4·x2 + N(0, 0.05) ──
n = 300
X = rng.normal(size=(n, 3))
y = 2.0 + 3.0 * X[:, 0] - 1.5 * X[:, 1] + 0.4 * X[:, 2] + rng.normal(0, 0.05, n)
Xt, Xv, yt, yv = split(X, y)

# collinear variant (ridge benefits): x3 = x0 + x1 + N(0,0.1)
Xc = np.column_stack([X, X[:, 0] + X[:, 1] + rng.normal(0, 0.1, n)])
Xct, Xcv, _, _ = split(Xc, y)

# sparse-truth variant (lasso/elastic_net): y = 1 + 2·x0 + 0·x1 − 1.2·x2
Xs = rng.normal(size=(n, 4))
ys = 1.0 + 2.0 * Xs[:, 0] + 0.0 * Xs[:, 1] - 1.2 * Xs[:, 2] + rng.normal(0, 0.05, n)
Xst, Xsv, yst, ysv = split(Xs, ys)

# outlier variant (robust): 15% of y corrupted by +25
Xo = rng.normal(size=(n, 2))
yo = 2.0 + 3.0 * Xo[:, 0] - 1.5 * Xo[:, 1] + rng.normal(0, 0.1, n)
flip = rng.random(n) < 0.15
yo[flip] += 25.0
Xot, Xov, yot, yov = split(Xo, yo)

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
)

# ── 1. ml_ols vs sklearn LinearRegression (coefficient-level proof) ──
res = ml_ols(con, yt, [Xt[:, 0], Xt[:, 1], Xt[:, 2]])
sk = LinearRegression().fit(Xt, yt)
ours_coef = res["coefficients"]  # [b1,b2,b3]
sk_coef = sk.coef_.tolist()
max_diff = max(abs(a - b) for a, b in zip(ours_coef, sk_coef))
intercept_diff = abs(res["intercept"] - sk.intercept_)
r2_diff = abs(res["r_squared"] - sk.score(Xt, yt))
report.add("ml_ols coefficient maxdiff", max_diff, 0.0, 1e-6, max_diff < 1e-6)
report.add("ml_ols intercept diff", intercept_diff, 0.0, 1e-6, intercept_diff < 1e-6)
report.add("ml_ols r2 diff vs sklearn", r2_diff, 0.0, 1e-6, r2_diff < 1e-6)

# ── 2. linear_regression vs sklearn (held-out predictions) ──
train(con, "lin01", "linear_regression", Xt, yt)
p_ours = np.asarray(predict(con, "lin01", Xv), float)
p_sk = sk.predict(Xv)
r2_ours, r2_sk = r2(yv, p_ours), r2(yv, p_sk)
report.add("lin R2(ours) on test", r2_ours, 0.99, 0.0, r2_ours >= 0.99,
           f"sklearn R2={r2_sk:.6f}")
report.add("lin pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 0.0001,
           sign_corr(p_ours, p_sk) >= 0.9999)

# ── 3. ridge_regression vs sklearn Ridge(alpha=1.0) on collinear data ──
train(con, "ridge01", "ridge_regression", Xct, yt, {"lambda": 1.0})
p_ours = np.asarray(predict(con, "ridge01", Xcv), float)
p_sk = Ridge(alpha=1.0).fit(Xct, yt).predict(Xcv)
r2_ours, r2_sk = r2(yv, p_ours), r2(yv, p_sk)
report.add("ridge R2(ours) on test", r2_ours, 0.99, 0.0, r2_ours >= 0.99,
           f"sklearn R2={r2_sk:.6f}")
report.add("ridge pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 0.001,
           sign_corr(p_ours, p_sk) >= 0.999)

# ── 4. lasso_regression vs sklearn Lasso (sparse truth) ──
train(con, "lasso01", "lasso_regression", Xst, yst, {"lambda": 0.02, "max_iter": 3000, "tol": 1e-6})
p_ours = np.asarray(predict(con, "lasso01", Xsv), float)
p_sk = Lasso(alpha=0.02, max_iter=5000).fit(Xst, yst).predict(Xsv)
r2_ours, r2_sk = r2(ysv, p_ours), r2(ysv, p_sk)
report.add("lasso R2(ours) on test", r2_ours, 0.95, 0.0, r2_ours >= 0.95,
           f"sklearn R2={r2_sk:.6f}")
report.add("lasso pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 0.01,
           sign_corr(p_ours, p_sk) >= 0.99)

# ── 5. elastic_net vs sklearn ElasticNet ──
train(con, "en01", "elastic_net", Xst, yst, {"alpha": 0.02, "l1_ratio": 0.5, "max_iter": 3000})
p_ours = np.asarray(predict(con, "en01", Xsv), float)
p_sk = ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=5000).fit(Xst, yst).predict(Xsv)
r2_ours, r2_sk = r2(ysv, p_ours), r2(ysv, p_sk)
report.add("elastic_net R2(ours) on test", r2_ours, 0.95, 0.0, r2_ours >= 0.95,
           f"sklearn R2={r2_sk:.6f}")
report.add("elastic_net pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 0.01,
           sign_corr(p_ours, p_sk) >= 0.99)

# ── 6. robust vs sklearn HuberRegressor on 15%-outlier data ──
# R² must be evaluated on the CLEAN test points (test outliers are +25 jumps
# that no model explains — sklearn Huber gets the same negative R² there).
train(con, "robust01", "robust", Xot, yot, {"c": 1.35, "max_iters": 50})
p_ours = np.asarray(predict(con, "robust01", Xov), float)
p_sk = HuberRegressor(epsilon=1.35).fit(Xot, yot).predict(Xov)
k = int(n * 0.7)
clean = ~flip[k:]
r2_ours_clean = r2(yov[clean], p_ours[clean])
r2_sk_clean = r2(yov[clean], p_sk[clean])
report.add("robust R2(ours) on clean test pts", r2_ours_clean, 0.9, 0.0,
           r2_ours_clean >= 0.9,
           f"sklearn Huber R2={r2_sk_clean:.6f} on clean pts")
report.add("robust pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 0.05,
           sign_corr(p_ours, p_sk) >= 0.95)

# ── 7. polynomial_regression vs sklearn PolynomialFeatures + LinearRegression ──
# Per-feature power expansion (no interaction terms — single feature ⇒ exact match).
from sklearn.preprocessing import PolynomialFeatures
Xpoly = np.sort(rng.uniform(-2, 2, 300))[:, None]
ypoly = (1.5 - 0.8 * Xpoly[:, 0] + 2.0 * Xpoly[:, 0] ** 2 - 0.3 * Xpoly[:, 0] ** 3
         + rng.normal(0, 0.05, 300))
Xpt, Xpv, ypt, ypv = split(Xpoly, ypoly)
train(con, "poly3", "polynomial_regression", Xpt, ypt, {"degree": 3})
p_ours = np.asarray(predict(con, "poly3", Xpv), float)
pf = PolynomialFeatures(3, include_bias=False)
p_sk = LinearRegression().fit(pf.fit_transform(Xpt), ypt).predict(pf.transform(Xpv))
r2_ours = r2(ypv, p_ours)
r2_sk = r2(ypv, p_sk)
report.add("polynomial(deg3) R2(ours)", r2_ours, 0.95, 0.0, r2_ours >= 0.95,
           f"sklearn R2={r2_sk:.6f}")
report.add("polynomial(deg3) pred corr vs sklearn", sign_corr(p_ours, p_sk), 1.0, 1e-9,
           sign_corr(p_ours, p_sk) >= 0.999999999, "single-feature ⇒ bitwise match")

ok = report.print_report()
con.close()
raise SystemExit(0 if ok else 1)
