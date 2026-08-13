"""Run the full duckdb-ml reproducibility suite.

    python labs/run_all.py

Runs every verify_*.py in its own subprocess (fresh interpreter + DuckDB
connection), plus a global determinism check: training the same model twice on
the same data must produce bit-identical predictions.

Exit code 0 = all checks passed; 1 = at least one check failed.
"""

import os
import subprocess
import sys

import numpy as np
from common import connect, train, predict, run_verify, REPO_ROOT

SCRIPTS = [
    ("verify_01_linear.py", "linear family (ols/linear/ridge/lasso/elastic_net/robust)"),
    ("verify_02_trees.py", "tree family (dt/rf/rf_classifier/xgboost gbdt)"),
    ("verify_03_kernel.py", "kernel family (svm/svr)"),
    ("verify_04_classify.py", "classification (logistic/multilogistic/ordinal/nb/knn)"),
    ("verify_05_unsup.py", "unsupervised (kmeans/dbscan/pca)"),
    ("verify_06_misc.py", "misc (mlp/cox/arima/ml_metrics/ml_cross_validate/assoc_rules)"),
]


def determinism_check():
    """Same data + same params twice → predictions must be bit-identical."""
    print("\n== determinism check ==")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 3))
    y = 2.0 + X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.1, 200)
    con = connect()
    train(con, "det_a", "linear_regression", X, y)
    p1 = predict(con, "det_a", X[:50])
    train(con, "det_a", "linear_regression", X, y)  # retrain, same data
    p2 = predict(con, "det_a", X[:50])
    same = p1 == p2
    print(f"{'retrain linear_regression identical':<44} {'—':>14} {'—':>14} {'—':>10}  "
          f"{'PASS' if same else 'FAIL'}")
    con.close()
    return same


def main():
    results = []
    ok = determinism_check()
    results.append(("determinism", ok))
    for script, desc in SCRIPTS:
        ok = run_verify(script, desc)
        results.append((script, ok))

    print("=" * 60)
    npass = sum(1 for _, ok in results if ok)
    nfail = len(results) - npass
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"-> {npass} groups passed / {nfail} failed")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
