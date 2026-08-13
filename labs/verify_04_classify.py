"""Classification family: logistic_regression / multilogistic / ordinal /
naive_bayes / knn_classifier / knn_regressor.

  - logistic/multilogistic/nb/knn : cross-checked vs scikit-learn
  - ordinal (cumulative logit)     : sklearn has no ordinal model — proven by
    accuracy vs the true ordered classes + rank correlation with the latent score
  - knn is an exact k-NN: agreement with sklearn should be ~100%
"""

import numpy as np
from scipy.stats import spearmanr
from common import (
    Report, connect, train, predict, r2, accuracy, sign_corr, SEED,
)

rng = np.random.default_rng(SEED)
report = Report("verify_04_classify")
con = connect()


def split(X, y, frac=0.7):
    k = int(len(y) * frac)
    return X[:k], X[k:], y[:k], y[k:]


# ── logistic: two overlapping blobs ──
Xl = np.vstack([rng.normal([-1.4, -1.4], 0.9, (200, 2)), rng.normal([1.4, 1.4], 0.9, (200, 2))])
yl = np.array([0.0] * 200 + [1.0] * 200)
Xlt, Xlv, ylt, ylv = split(Xl, yl)

# ── multilogistic: three blobs ──
Xm = np.vstack([
    rng.normal([-2, -2], 0.9, (150, 2)),
    rng.normal([2, -2], 0.9, (150, 2)),
    rng.normal([0, 2.5], 0.9, (150, 2)),
])
ym = np.array([0.0] * 150 + [1.0] * 150 + [2.0] * 150)
Xmt, Xmv, ymt, ymv = split(Xm, ym)

# ── ordinal: latent score -> 4 ordered classes ──
n = 400
Xo = rng.normal(size=(n, 2))
latent = 1.5 * Xo[:, 0] - 1.0 * Xo[:, 1] + rng.normal(0, 0.5, n)
yo = np.digitize(latent, [-1.0, 0.0, 1.0]).astype(float)  # 0..3
Xot, Xov, yot, yov = split(Xo, yo)
k = int(len(latent) * 0.7)
lt, lv = latent[:k], latent[k:]

# ── naive bayes: two Gaussian blobs, independent features ──
Xn = np.vstack([rng.normal([-1, -1], 0.8, (200, 2)), rng.normal([1, 1], 0.8, (200, 2))])
yn = np.array([0.0] * 200 + [1.0] * 200)
Xnt, Xnv, ynt, ynv = split(Xn, yn)

# ── knn: smooth surface (regressor) + three blobs (classifier) ──
Xk = rng.uniform(-3, 3, (300, 2))
yk = Xk[:, 0] ** 2 + Xk[:, 1] ** 2 + rng.normal(0, 0.1, 300)
Xkt, Xkv, ykt, ykv = split(Xk, yk)
Xkc = np.vstack([
    rng.normal([-2, -2], 0.5, (150, 2)),
    rng.normal([2, -2], 0.5, (150, 2)),
    rng.normal([0, 2], 0.5, (150, 2)),
])
ykc = np.array([0.0] * 150 + [1.0] * 150 + [2.0] * 150)
Xkct, Xkcv, ykct, ykcv = split(Xkc, ykc)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# ── 1. logistic_regression vs sklearn ──
train(con, "log01", "logistic_regression", Xlt, ylt, {"lr": 0.05, "epochs": 500})
p_ours = np.asarray(predict(con, "log01", Xlv), float).round()
p_sk = LogisticRegression(max_iter=2000).fit(Xlt, ylt).predict(Xlv)
acc_ours, acc_sk = accuracy(ylv, p_ours), accuracy(ylv, p_sk)
report.add("logistic accuracy(ours)", acc_ours, 0.92, 0.0, acc_ours >= 0.92,
           f"sklearn acc={acc_sk:.4f}")
report.add("logistic acc diff vs sklearn", abs(acc_ours - acc_sk), 0.0, 0.02,
           abs(acc_ours - acc_sk) <= 0.02)

# ── 2. multilogistic vs sklearn multinomial ──
train(con, "multi01", "multilogistic", Xmt, ymt, {"lr": 0.1, "max_epochs": 1500})
p_ours = np.asarray(predict(con, "multi01", Xmv), float).round()
p_sk = LogisticRegression(max_iter=2000).fit(Xmt, ymt).predict(Xmv)
acc_ours, acc_sk = accuracy(ymv, p_ours), accuracy(ymv, p_sk)
report.add("multilogistic accuracy(ours)", acc_ours, 0.88, 0.0, acc_ours >= 0.88,
           f"sklearn acc={acc_sk:.4f}")
report.add("multilogistic acc diff vs sklearn", abs(acc_ours - acc_sk), 0.0, 0.03,
           abs(acc_ours - acc_sk) <= 0.03)

# ── 3. ordinal: no sklearn equivalent — accuracy + latent monotonicity ──
train(con, "ord01", "ordinal", Xot, yot, {"lr": 0.1, "max_epochs": 2000})
p_ours = np.asarray(predict(con, "ord01", Xov), float).round()
acc_ours = accuracy(yov, p_ours)
rho = spearmanr(p_ours, lv).statistic
report.add("ordinal accuracy vs true classes", acc_ours, 0.78, 0.0, acc_ours >= 0.78,
           "no sklearn reference (cumulative logit)")
report.add("ordinal pred~latent spearman", rho, 0.9, 0.0, rho >= 0.9,
           "ordered classes must track latent score monotonically")

# ── 4. naive_bayes vs sklearn GaussianNB ──
train(con, "nb01", "naive_bayes", Xnt, ynt)
p_ours = np.asarray(predict(con, "nb01", Xnv), float).round()
p_sk = GaussianNB().fit(Xnt, ynt).predict(Xnv)
acc_ours, acc_sk = accuracy(ynv, p_ours), accuracy(ynv, p_sk)
report.add("naive_bayes accuracy(ours)", acc_ours, 0.85, 0.0, acc_ours >= 0.85,
           f"sklearn acc={acc_sk:.4f}")
report.add("naive_bayes acc diff vs sklearn", abs(acc_ours - acc_sk), 0.0, 0.05,
           abs(acc_ours - acc_sk) <= 0.05)

# ── 5. knn_classifier: exact k-NN, agreement vs sklearn ──
train(con, "knnc01", "knn_classifier", Xkct, ykct, {"k": 5})
p_ours = np.asarray(predict(con, "knnc01", Xkcv), float).round()
p_sk = KNeighborsClassifier(n_neighbors=5).fit(Xkct, ykct).predict(Xkcv)
agree = accuracy(p_ours, p_sk)
acc_ours, acc_sk = accuracy(ykcv, p_ours), accuracy(ykcv, p_sk)
report.add("knn_classifier accuracy(ours)", acc_ours, 0.95, 0.0, acc_ours >= 0.95,
           f"sklearn acc={acc_sk:.4f}")
report.add("knn_classifier agreement vs sklearn", agree, 0.98, 0.0, agree >= 0.98,
           "both are exact k-NN; labels should agree")

# ── 6. knn_regressor: exact k-NN ──
train(con, "knnr01", "knn_regressor", Xkt, ykt, {"k": 5})
p_ours = np.asarray(predict(con, "knnr01", Xkv), float)
p_sk = KNeighborsRegressor(n_neighbors=5).fit(Xkt, ykt).predict(Xkv)
r2_ours, r2_sk = r2(ykv, p_ours), r2(ykv, p_sk)
maxdiff = float(np.max(np.abs(p_ours - p_sk)))
report.add("knn_regressor R2(ours)", r2_ours, 0.9, 0.0, r2_ours >= 0.9,
           f"k=5 on noisy quadratic surface; sklearn R2={r2_sk:.6f}")
report.add("knn_regressor max|pred diff| vs sklearn", maxdiff, 0.0, 1e-6,
           maxdiff < 1e-6, "exact k-NN should be numerically identical")

ok = report.print_report()
con.close()
raise SystemExit(0 if ok else 1)
