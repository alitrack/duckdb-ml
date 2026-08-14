"""Misc family: mlp_regressor / cox / arima + pipeline helpers (ml_metrics,
ml_cross_validate, ml_assoc_rules).

  - mlp_regressor : vs sklearn MLPRegressor (loose tolerance — SGD variance)
  - cox           : no sklearn equivalent (lifelines) — Spearman(risk score,
                    true log-hazard) on synthetic survival data
  - arima         : AR(2) closed-form reference — forecast correlation + MSE
  - ml_metrics    : exact match vs sklearn.metrics (deterministic formulas)
  - ml_cross_validate : mean R² vs sklearn cross_val_score
  - ml_assoc_rules: brute-force support/confidence/lift check on fixed data
"""

import json

import numpy as np
from scipy.stats import spearmanr
from common import (
    Report, connect, train, predict, r2, accuracy, SEED,
)

rng = np.random.default_rng(SEED)
report = Report("verify_06_misc")
con = connect()


def split(X, y, frac=0.7):
    k = int(len(y) * frac)
    return X[:k], X[k:], y[:k], y[k:]


# ── mlp: y = sin(x0)·cos(x1) ──
Xm = rng.uniform(-3, 3, (400, 2))
ym = np.sin(Xm[:, 0]) * np.cos(Xm[:, 1]) + rng.normal(0, 0.05, 400)
Xmt, Xmv, ymt, ymv = split(Xm, ym)

# ── cox: exponential survival, log-hazard = 0.8·x0 − 0.5·x1 ──
n = 400
Xc = rng.normal(size=(n, 2))
log_h = 0.8 * Xc[:, 0] - 0.5 * Xc[:, 1]
t = rng.exponential(1.0 / np.exp(log_h))
cens = rng.random(n) < 0.2
t[cens] = rng.uniform(0, t[cens])  # censored before event
event = (~cens).astype(float)
Xct, Xcv, _, _ = split(Xc, t)  # same split for X; we'll use all for scoring

# ── arima: y_t = 0.6·y_{t-1} − 0.3·y_{t-2} + N(0, 0.1) ──
na = 220
y_ar = np.zeros(na)
for i in range(2, na):
    y_ar[i] = 0.6 * y_ar[i - 1] - 0.3 * y_ar[i - 2] + rng.normal(0, 0.1)
y_train = y_ar[:150]
y_future = y_ar[150:170]  # 20 true future values

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# ── 1. mlp_regressor vs sklearn MLPRegressor (loose) ──
train(con, "mlp01", "mlp_regressor",
      Xmt, ymt, {"hidden_size": 12, "lr": 0.02, "momentum": 0.9, "iterations": 600, "batch_size": 32})
p_ours = np.asarray(predict(con, "mlp01", Xmv), float)
p_sk = MLPRegressor(
    hidden_layer_sizes=(12,), alpha=0.0, learning_rate_init=0.02, momentum=0.9,
    max_iter=600, batch_size=32, random_state=SEED,
).fit(Xmt, ymt).predict(Xmv)
r2_ours, r2_sk = r2(ymv, p_ours), r2(ymv, p_sk)
report.add("mlp R2(ours) on test", r2_ours, 0.85, 0.0, r2_ours >= 0.85,
           f"sklearn R2={r2_sk:.6f} (SGD variance, loose tolerance)")
report.add("mlp R2 diff vs sklearn", abs(r2_ours - r2_sk), 0.0, 0.15,
           abs(r2_ours - r2_sk) <= 0.15)

# ── 2. cox: risk score must track true log-hazard ──
con.execute(
    "SELECT ml_cox_train(?, ?, ?, ?, ?)",
    ["cox01",
     json.dumps([float(v) for v in t]),
     json.dumps([float(v) for v in event]),
     json.dumps([list(map(float, r)) for r in Xc]),
     "{}"],
).fetchone()
p_ours = np.asarray(predict(con, "cox01", Xc), float)
rho = spearmanr(p_ours, log_h).statistic
report.add("cox risk~log-hazard spearman", rho, 0.9, 0.0, rho >= 0.9,
           "higher predicted risk ⇔ higher true hazard")

# ── 2b. kaplan-meier: median survival vs closed-form product-limit ──
# No library needed — the product-limit estimator is a closed formula; verify
# bit-exact median against a numpy reimplementation of the same formula.
from common import predict as _predict
nk = 200
kt = np.sort(rng.exponential(2.0, nk).round(1))
ke = (rng.random(nk) > 0.3).astype(float)
kt[ke == 0] += rng.uniform(0.0, 0.5, int((ke == 0).sum()))
con.execute("SELECT ml_km_train(?, ?, ?)",
            ["km01", json.dumps(kt.tolist()), json.dumps(ke.tolist())]).fetchone()
med_ours = float(_predict(con, "km01", np.zeros((3, 1)))[0])
S_km, cur = [], 1.0
for t in np.unique(kt[ke == 1]):
    nrisk = int((kt >= t).sum())
    d = int(((kt == t) & (ke == 1)).sum())
    cur *= 1.0 - d / nrisk
    S_km.append((t, cur))
med_ref = next((t for t, s in S_km if s <= 0.5), float(kt.max()))
report.add("kaplan-meier median == product-limit ref", int(med_ours == med_ref),
           1.0, 0.0, med_ours == med_ref,
           f"numpy closed-form median={med_ref}")

# ── 3. arima: AR(2) forecast vs true future ──
# Deterministic AR(2) series (y_t = 0.6·y_{t-1} + 0.25·y_{t-2}, no noise):
# a perfectly predictable series must be predicted near-exactly. This is the
# strongest form of the check — earlier noise-dominated data (σ=0.05 vs AR
# signal of the same scale) made even a correct AR forecast look random,
# because the series genuinely has nothing to predict after a few steps.
s_ar = [1.0, 1.5]
for _ in range(300):
    s_ar.append(0.6 * s_ar[-1] + 0.25 * s_ar[-2])
n_ar = len(s_ar)
y_train = np.array(s_ar[: n_ar - 10])
y_future = np.array(s_ar[n_ar - 10 :])
train(con, "arima01", "arima", np.zeros((len(y_train), 1)), y_train,
      {"p": 2, "d": 0, "q": 0})
h = np.arange(1, 11, dtype=float)[:, None]
p_ours = np.asarray(predict(con, "arima01", h), float)
corr_ar = float(np.corrcoef(p_ours, y_future)[0, 1])
nrmse_ar = float(np.sqrt(np.mean((p_ours - y_future) ** 2)) / np.std(y_future))
report.add("arima forecast corr vs true", corr_ar, 0.99, 0.0, corr_ar >= 0.99,
           "deterministic AR(2), 10-step")
report.add("arima forecast NRMSE", nrmse_ar, 0.5, 0.0, nrmse_ar <= 0.5,
           "normalized by series std")

# ── 4. ml_metrics vs sklearn.metrics (exact) ──
y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.7, 0.6, 0.3, 0.8, 0.45, 0.95])
m = json.loads(con.execute(
    "SELECT ml_metrics(?, ?, 'binary')",
    [json.dumps(y_true.tolist()), json.dumps(y_pred.tolist())],
).fetchone()[0])
checks = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1": f1_score(y_true, y_pred, zero_division=0),
}
for name, ref in checks.items():
    if name in m:
        report.add(f"ml_metrics.{name} diff", abs(m[name] - ref), 0.0, 1e-9,
                   abs(m[name] - ref) < 1e-9, f"ours={m[name]:.9f} sklearn={ref:.9f}")
    else:
        report.add(f"ml_metrics.{name} present", False, True, 0.0, False, "key missing")
# regression metrics via auto task
yr_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
yr_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.2])
mr = json.loads(con.execute(
    "SELECT ml_metrics(?, ?)",
    [json.dumps(yr_true.tolist()), json.dumps(yr_pred.tolist())],
).fetchone()[0])
for name, ref in {
    "mse": mean_squared_error(yr_true, yr_pred),
    "rmse": root_mean_squared_error(yr_true, yr_pred),
    "mae": mean_absolute_error(yr_true, yr_pred),
    "r2": r2_score(yr_true, yr_pred),
}.items():
    if name in mr:
        report.add(f"ml_metrics.{name} diff", abs(mr[name] - ref), 0.0, 1e-9,
                   abs(mr[name] - ref) < 1e-9, f"ours={mr[name]:.9f} sklearn={ref:.9f}")
    else:
        report.add(f"ml_metrics.{name} present", False, True, 0.0, False, "key missing")

# ── 5. ml_cross_validate vs sklearn cross_val_score ──
Xcv2 = rng.normal(size=(200, 2))
ycv2 = 1.5 * Xcv2[:, 0] - 0.8 * Xcv2[:, 1] + rng.normal(0, 0.1, 200)
cv = json.loads(con.execute(
    "SELECT ml_cross_validate(?, ?, ?, ?, ?)",
    ["linear_regression",
     json.dumps([list(map(float, r)) for r in Xcv2]),
     json.dumps([float(v) for v in ycv2]),
     "{}", "5"],
).fetchone()[0])
mean_r2_ours = float(cv.get("mean_r2", float("nan")))
sk_scores = cross_val_score(LinearRegression(), Xcv2, ycv2, cv=5)
report.add("ml_cross_validate mean_r2 diff", abs(mean_r2_ours - sk_scores.mean()),
           0.0, 0.05, abs(mean_r2_ours - sk_scores.mean()) <= 0.05,
           f"sklearn CV mean={sk_scores.mean():.6f} folds={len(cv.get('folds', []))}")

# ── 6. ml_assoc_rules vs brute-force Apriori check ──
txns = [
    {"tid": 1, "items": ["a", "b", "c"]},
    {"tid": 2, "items": ["a", "b"]},
    {"tid": 3, "items": ["a", "c"]},
    {"tid": 4, "items": ["b", "c"]},
    {"tid": 5, "items": ["a", "b", "c", "d"]},
    {"tid": 6, "items": ["b", "d"]},
]
n_txn = len(txns)
rules = json.loads(con.execute(
    "SELECT ml_assoc_rules(?, ?, ?)",
    [json.dumps(txns), 0.3, 0.5],
).fetchone()[0])

def brute(ante, cons):
    supp_ab = sum(1 for t in txns if set(ante) <= set(t["items"]) and set(cons) <= set(t["items"])) / n_txn
    supp_a = sum(1 for t in txns if set(ante) <= set(t["items"])) / n_txn
    conf = supp_ab / supp_a if supp_a > 0 else 0.0
    supp_b = sum(1 for t in txns if set(cons) <= set(t["items"])) / n_txn
    lift = conf / supp_b if supp_b > 0 else 0.0
    return supp_ab, conf, lift

worst = 0.0
n_rules = 0
for r in rules["rules"]:
    ante = r.get("antecedent", [])
    cons = r.get("consequent", [])
    b_sup, b_conf, b_lift = brute(ante, cons)
    d = max(
        abs(r.get("support", -1) - b_sup),
        abs(r.get("confidence", -1) - b_conf),
        abs(r.get("lift", -1) - b_lift),
    )
    worst = max(worst, d)
    n_rules += 1
report.add("assoc_rules #rules found", n_rules, 3, 0.0, n_rules >= 3,
           "min_support=0.3 min_confidence=0.5")
report.add("assoc_rules worst metric diff vs brute-force", worst, 0.0, 1e-9,
           worst < 1e-9, f"checked {n_rules} rules × (support/confidence/lift)")

ok = report.print_report()
con.close()
raise SystemExit(0 if ok else 1)
