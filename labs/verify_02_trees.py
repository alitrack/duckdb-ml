"""Tree family: decision_tree / random_forest / rf_classifier / xgboost_regression /
xgboost_binary — cross-checked against scikit-learn.

NOTE: split search and bootstrap sampling differ between the hand-written Rust
implementation and sklearn, so tolerances are on predictive quality (R² /
accuracy), not bit-identical trees. rf_classifier also exercises the
string-label encode/decode path end-to-end.
"""

import numpy as np
from common import (
    Report, connect, train, predict, r2, accuracy, SEED,
)

rng = np.random.default_rng(SEED)
report = Report("verify_02_trees")
con = connect()


def split(X, y, frac=0.7):
    k = int(len(y) * frac)
    return X[:k], X[k:], y[:k], y[k:]


# regression data: threshold rules (trees' home turf)
#   y = 3·[x0>0] − 2·[x1<0] + 0.5·x2 + N(0,0.1)
n = 400
X = rng.normal(size=(n, 3))
y = 3.0 * (X[:, 0] > 0) - 2.0 * (X[:, 1] < 0) + 0.5 * X[:, 2] + rng.normal(0, 0.1, n)
Xt, Xv, yt, yv = split(X, y)

# classification data: annulus with a clear gap — radius<1.0 -> 1, >1.8 -> 0
n = 500
Xc = rng.uniform(-3, 3, size=(n, 2))
r = np.sqrt((Xc ** 2).sum(1))
yc = np.where(r < 1.0, 1.0, np.where(r > 1.8, 0.0, 0.5))  # 0.5 = excluded gap
keep = yc != 0.5
Xc, yc = Xc[keep], yc[keep]
Xct, Xcv, yct, ycv = split(Xc, yc)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier

# ── 1. decision_tree vs sklearn ──
dt_params = {"max_depth": 8, "min_samples_split": 5, "min_samples_leaf": 2}
train(con, "dt01", "decision_tree", Xt, yt, dt_params)
p_ours = np.asarray(predict(con, "dt01", Xv), float)
p_sk = DecisionTreeRegressor(
    max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=SEED
).fit(Xt, yt).predict(Xv)
r2_ours, r2_sk = r2(yv, p_ours), r2(yv, p_sk)
report.add("dt R2(ours) on test", r2_ours, 0.95, 0.0, r2_ours >= 0.95,
           f"sklearn R2={r2_sk:.6f}")
report.add("dt R2 diff vs sklearn", abs(r2_ours - r2_sk), 0.0, 0.05,
           abs(r2_ours - r2_sk) <= 0.05)

# ── 2. random_forest vs sklearn ──
rf_params = {"n_estimators": 50, "max_depth": 8}
train(con, "rf01", "random_forest", Xt, yt, rf_params)
p_ours = np.asarray(predict(con, "rf01", Xv), float)
p_sk = RandomForestRegressor(
    n_estimators=50, max_depth=8, min_samples_split=2, min_samples_leaf=1,
    random_state=SEED, n_jobs=-1,
).fit(Xt, yt).predict(Xv)
r2_ours, r2_sk = r2(yv, p_ours), r2(yv, p_sk)
report.add("rf R2(ours) on test", r2_ours, 0.95, 0.0, r2_ours >= 0.95,
           f"sklearn R2={r2_sk:.6f}")
report.add("rf R2 diff vs sklearn", abs(r2_ours - r2_sk), 0.0, 0.05,
           abs(r2_ours - r2_sk) <= 0.05)

# ── 3. rf_classifier (numeric labels) vs sklearn ──
train(con, "rfc01", "rf_classifier", Xct, yct, rf_params)
p_ours = np.asarray(predict(con, "rfc01", Xcv), float).round()
p_sk = RandomForestClassifier(
    n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1,
).fit(Xct, yct).predict(Xcv)
acc_ours, acc_sk = accuracy(ycv, p_ours), accuracy(ycv, p_sk)
report.add("rfc accuracy(ours)", acc_ours, 0.99, 0.0, acc_ours >= 0.99,
           f"sklearn acc={acc_sk:.4f}")
report.add("rfc acc diff vs sklearn", abs(acc_ours - acc_sk), 0.0, 0.01,
           abs(acc_ours - acc_sk) <= 0.01)

# ── 4. rf_classifier with STRING labels (encode/decode path) ──
labels = np.where(yc > 0.5, "inside", "outside")
yt_s, yv_s = labels[: len(yct)], labels[len(yct):]
train(con, "rfc_str", "rf_classifier", Xct, yt_s, rf_params, y_is_str=True)
p_str = predict(con, "rfc_str", Xcv)  # must decode back to strings
is_str = all(isinstance(v, str) for v in p_str)
acc_str = accuracy(yv_s, p_str)
report.add("rfc string-label decode", is_str, True, 0.0, is_str,
           f"sample={p_str[:3]}")
report.add("rfc string-label accuracy", acc_str, 0.99, 0.0, acc_str >= 0.99,
           f"sklearn acc={acc_sk:.4f}")

# ── 5. xgboost_regression (in-DB GBDT) vs sklearn GradientBoostingRegressor ──
xgb_params = {"n_estimators": 80, "learning_rate": 0.1, "max_depth": 4}
train(con, "xgb01", "xgboost_regression", Xt, yt, xgb_params)
p_ours = np.asarray(predict(con, "xgb01", Xv), float)
p_sk = GradientBoostingRegressor(
    n_estimators=80, learning_rate=0.1, max_depth=4, random_state=SEED,
).fit(Xt, yt).predict(Xv)
r2_ours, r2_sk = r2(yv, p_ours), r2(yv, p_sk)
report.add("xgboost_reg R2(ours)", r2_ours, 0.95, 0.0, r2_ours >= 0.95,
           f"sklearn GBR R2={r2_sk:.6f}")
report.add("xgboost_reg R2 diff vs sklearn", abs(r2_ours - r2_sk), 0.0, 0.03,
           abs(r2_ours - r2_sk) <= 0.03)

# ── 6. xgboost_binary vs sklearn GradientBoostingClassifier ──
train(con, "xgbbin", "xgboost_binary", Xct, yct, xgb_params)
p_ours = np.asarray(predict(con, "xgbbin", Xcv), float).round()
p_sk = GradientBoostingClassifier(
    n_estimators=80, learning_rate=0.1, max_depth=4, random_state=SEED,
).fit(Xct, yct).predict(Xcv)
acc_ours, acc_sk = accuracy(ycv, p_ours), accuracy(ycv, p_sk)
report.add("xgboost_bin accuracy(ours)", acc_ours, 0.97, 0.0, acc_ours >= 0.97,
           f"sklearn GBC acc={acc_sk:.4f}")
report.add("xgboost_bin acc diff vs sklearn", abs(acc_ours - acc_sk), 0.0, 0.03,
           abs(acc_ours - acc_sk) <= 0.03)

ok = report.print_report()
con.close()
raise SystemExit(0 if ok else 1)
