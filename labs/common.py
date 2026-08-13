"""Shared infrastructure for duckdb-ml reproducibility labs.

Purpose: prove that the hand-written Rust algorithms in duckdb-ml are reliable
by (1) determinism checks, (2) cross-validation against reference
implementations (scikit-learn / statsmodels) on seeded synthetic data, and
(3) explicit PASS/FAIL tolerances that are printed with every check.

Usage (from repo root):
    uv run --with duckdb --with scikit-learn --with scipy --with statsmodels python labs/run_all.py
or:
    pip install -r labs/requirements.txt && python labs/run_all.py

Every dataset is generated with a fixed seed (np.random.default_rng(SEED)), so
two runs of this suite produce identical numbers.
"""

import json
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_PATH = os.path.join(REPO_ROOT, "build", "release", "ml.duckdb_extension")
SEED = 42


# ───────────────────────────── DuckDB session ─────────────────────────────

def connect():
    import duckdb

    if not os.path.exists(EXT_PATH):
        sys.exit(f"extension not found: {EXT_PATH} — run `make release` first")
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{EXT_PATH}'")
    return con


def _load_table(con, tbl, X, y=None, y_is_str=False):
    """Create (or replace) table `tbl` with columns f0..fd-1 and (optionally) y."""
    d = X.shape[1]
    cols = [f"f{i} DOUBLE" for i in range(d)]
    if y is not None:
        cols.append("y " + ("VARCHAR" if y_is_str else "DOUBLE"))
    con.execute(f"CREATE OR REPLACE TABLE {tbl} ({', '.join(cols)})")
    rows = [list(map(float, x)) for x in X]
    if y is not None:
        for r, yi in zip(rows, y):
            r.append(yi if y_is_str else float(yi))
    placeholders = ", ".join(["?"] * len(rows[0]))
    con.executemany(f"INSERT INTO {tbl} VALUES ({placeholders})", rows)


def train(con, model_name, algo, X, y, params=None, y_is_str=False):
    """Train via the ml_train_model scalar. X:(n,d) float, y:(n,) numeric or str."""
    tbl = f"labs_{model_name.replace('-', '_')}"
    _load_table(con, tbl, X, y, y_is_str)
    feat_cols = ", ".join(f"f{i}" for i in range(X.shape[1]))
    sql = (
        f"SELECT ml_train_model(?, ?, "
        f"(SELECT to_json(list(y)) FROM {tbl}), "
        f"(SELECT to_json(list([{feat_cols}])) FROM {tbl}), ?)"
    )
    con.execute(sql, [model_name, algo, json.dumps(params or {})]).fetchone()
    return model_name


def predict(con, model_name, X):
    """Batch predict via ml_predict_batch_value; returns list of float or str."""
    tbl = f"labs_pred_{model_name.replace('-', '_')}"
    _load_table(con, tbl, X)
    feat_cols = ", ".join(f"f{i}" for i in range(X.shape[1]))
    sql = (
        f"SELECT ml_predict_batch_value(?, "
        f"(SELECT to_json(list([{feat_cols}])) FROM {tbl}))"
    )
    raw = con.execute(sql, [model_name]).fetchone()[0]
    return json.loads(raw)


def ml_ols(con, y, Xs):
    """ml_ols(actual_json, feat_json...) -> dict with coefficients/intercept/r2."""
    sql = "SELECT ml_ols(" + ", ".join(["?"] * (1 + len(Xs))) + ")"
    args = [json.dumps(list(map(float, y)))] + [json.dumps(list(map(float, c))) for c in Xs]
    return json.loads(con.execute(sql, args).fetchone()[0])


# ───────────────────────────── metrics ─────────────────────────────

def r2(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def cluster_alignment(labels_ours, labels_ref, n_clusters):
    """Fraction of points where our label maps to the reference label via the
    best (Hungarian) cluster-to-cluster assignment. Noise (-1) must match."""
    from scipy.optimize import linear_sum_assignment

    ours, ref = np.asarray(labels_ours), np.asarray(labels_ref)
    # separate noise handling: -1 must match -1
    noise_ok = np.mean((ours == -1) == (ref == -1))
    mask = (ours != -1) & (ref != -1)
    if mask.sum() == 0:
        return noise_ok
    # build confusion matrix over k clusters (treat each cluster id separately)
    uo = sorted(set(ours[mask]))
    ur = sorted(set(ref[mask]))
    conf = np.zeros((len(uo), len(ur)))
    for i, a in enumerate(uo):
        for j, b in enumerate(ur):
            conf[i, j] = np.sum((ours[mask] == a) & (ref[mask] == b))
    ri, ci = linear_sum_assignment(-conf)
    aligned = conf[ri, ci].sum()
    return float((aligned + np.sum((ours == -1) & (ref == -1))) / len(ours))


def sign_corr(a, b):
    """|Pearson correlation| — for PCA components (sign is arbitrary)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return abs(float(np.corrcoef(a, b)[0, 1]))


# ───────────────────────────── report ─────────────────────────────

class Report:
    def __init__(self, title):
        self.title = title
        self.checks = []

    def add(self, name, ours, ref, tol, passed, detail=""):
        self.checks.append(
            {
                "name": name,
                "ours": ours,
                "ref": ref,
                "tol": tol,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    def print_report(self):
        npass = sum(1 for c in self.checks if c["passed"])
        nfail = len(self.checks) - npass
        print(f"\n== {self.title} ==")
        print(f"{'check':<44} {'ours':>14} {'ref':>14} {'tol':>10}  result")
        print("-" * 88)
        for c in self.checks:
            mark = "PASS" if c["passed"] else "FAIL"
            print(
                f"{c['name']:<44} {c['ours']:>14.6g} {c['ref']:>14.6g} "
                f"{c['tol']:>10.3g}  {mark}  {c['detail']}"
            )
        print(f"-> {npass} passed / {nfail} failed\n")
        return nfail == 0


def run_verify(script, title):
    """Run one verify_*.py in a subprocess (fresh interpreter, own duckdb conn)."""
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, os.path.join(here, script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
    return proc.returncode == 0
