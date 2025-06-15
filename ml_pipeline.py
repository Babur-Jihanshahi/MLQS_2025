"""
Transport-mode ML pipeline – v2.2
================================
• 5 random seeds × 5-fold CV       • Optional GridSearch tuning (Table 7.4 grids)
• Detailed table (mean ± sd for Train/Test Acc & macro-F1 + CV-F1) – like the book
• Best-model hyper-parameters dump  • Top-feature list for tree models
• Error-bar plot for Test accuracy ± 2 sd
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")   # keep console tidy

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset(csv: Union[str, Path]):
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    X = df.drop(columns=[c for c in ("label", "timestamp", "Unnamed: 0") if c in df])
    X = X.fillna(method="ffill").fillna(method="bfill")
    y = df["label"]
    enc = LabelEncoder().fit(y)
    return X, enc.transform(y), enc


def split(X, y, seed: int, test_size: float = 0.3):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def preprocessor(X):
    num_cols = X.select_dtypes(["float", "int"]).columns.tolist()
    return ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="passthrough")


def make_pipe(clf, prep, k: int):
    return Pipeline([
        ("prep",   prep),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("clf",    clf),
    ])

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Models & hyper-parameter grids  (Table 7.4)
# ──────────────────────────────────────────────────────────────────────────────
def models(seed: int):
    return {
        "NN":  MLPClassifier(max_iter=2000, early_stopping=True, random_state=seed),
        "SVM": SVC(probability=True, max_iter=2000, random_state=seed),
        "KNN": KNeighborsClassifier(),
        "DT":  DecisionTreeClassifier(random_state=seed),
        "NB":  GaussianNB(),
        "RF":  RandomForestClassifier(n_jobs=-1, random_state=seed),
    }


def grids() -> Dict[str, Dict[str, List]]:
    return {
        "NN":  {"clf__hidden_layer_sizes": [(5,), (25,), (100,)],
                "clf__activation": ["relu", "logistic"]},
        "SVM": {"clf__C": [1, 10], "clf__tol": [1e-3, 1e-4],
                "clf__kernel": ["rbf", "poly"]},
        "KNN": {"clf__n_neighbors": [1, 5, 10]},
        "DT":  {"clf__min_samples_leaf": [2, 20, 100],
                "clf__criterion": ["gini", "entropy"]},
        "NB":  {},
        "RF":  {"clf__n_estimators": [50, 100],
                "clf__min_samples_leaf": [2, 20],
                "clf__criterion": ["gini", "entropy"]},
    }

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _feat_names(pipe, X):
    prep = pipe.named_steps["prep"]
    try:
        names = prep.get_feature_names_out()
    except AttributeError:              # < sklearn-1.1 fallback
        names = np.array([f"F{i}" for i in range(prep.transform(X.iloc[[0]]).shape[1])])
    mask = pipe.named_steps["select"].get_support()
    return names[mask]


def _top_importances(pipe, X, k=5):
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    names = _feat_names(pipe, X)
    imps  = clf.feature_importances_
    idx   = np.argsort(imps)[-k:][::-1]
    return [(names[i], float(imps[i])) for i in idx]

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Single run  (optionally with tuning)
# ──────────────────────────────────────────────────────────────────────────────
def run_once(Xtr, ytr, Xte, yte, *, seed: int, k: int, tune: bool):
    prep = preprocessor(Xtr)
    cv   = StratifiedKFold(5, shuffle=True, random_state=seed)
    out  = {}

    for name, base in models(seed).items():
        pipe        = make_pipe(clone(base), prep, k)
        best_params = None

        if tune and grids()[name]:
            gs = GridSearchCV(pipe, grids()[name], cv=cv,
                              scoring="f1_macro", n_jobs=-1, verbose=0)
            gs.fit(Xtr, ytr)
            pipe, best_params = gs.best_estimator_, gs.best_params_
        else:
            pipe.fit(Xtr, ytr)

        yhat_tr, yhat_te = pipe.predict(Xtr), pipe.predict(Xte)
        cv_scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro")

        out[name] = {
            "train_acc": accuracy_score(ytr, yhat_tr),
            "test_acc":  accuracy_score(yte, yhat_te),
            "train_f1":  f1_score(ytr, yhat_tr, average="macro"),
            "test_f1":   f1_score(yte, yhat_te, average="macro"),
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "imp":       _top_importances(pipe, Xtr),
            "params":    best_params,
        }
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Multi-run pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(csv="data/final_data_with_patterns.csv",
                 *, k_best=15, runs=5, tune=False):

    X, y, _   = load_dataset(csv)
    all_runs  = []
    for s in range(runs):
        Xtr, Xte, ytr, yte = split(X, y, s)
        all_runs.append(run_once(Xtr, ytr, Xte, yte, seed=s, k=k_best, tune=tune))
        print(f"run {s+1}/{runs} done")

    # ─── Detailed table (mean ± sd) ─────────────────────────────────────────
    rows = []
    for m in all_runs[0]:
        tr_acc = np.array([r[m]["train_acc"] for r in all_runs])
        te_acc = np.array([r[m]["test_acc"]  for r in all_runs])
        tr_f1  = np.array([r[m]["train_f1"]  for r in all_runs])
        te_f1  = np.array([r[m]["test_f1"]   for r in all_runs])
        cv_mu  = np.array([r[m]["cv_mean"]   for r in all_runs])
        cv_sd  = np.array([r[m]["cv_std"]    for r in all_runs])

        rows.append({
            "Model":      m,
            "Train-Acc": f"{tr_acc.mean():.4f} ±{tr_acc.std():.4f}",
            "Test-Acc":  f"{te_acc.mean():.4f} ±{te_acc.std():.4f}",
            "Train-F1":  f"{tr_f1.mean():.4f} ±{tr_f1.std():.4f}",
            "Test-F1":   f"{te_f1.mean():.4f} ±{te_f1.std():.4f}",
            "CV-F1":     f"{cv_mu.mean():.4f} ±{cv_sd.mean():.4f}",
        })

    detailed = pd.DataFrame(rows)
    detailed["_sort"] = detailed["Test-F1"].str.split().str[0].astype(float)
    detailed = detailed.sort_values("_sort", ascending=False).drop(columns="_sort")

    print("\n=== Detailed performance (mean ± sd) ===")
    print(detailed.to_string(index=False))

    # ─── Best model & params (highest mean Test-F1) ────────────────────────
    best_name = detailed.iloc[0]["Model"]
    best_run  = max(all_runs, key=lambda d: d[best_name]["test_f1"])
    if best_run[best_name]["params"]:
        print(f"\n>>> best model: {best_name}")
        for k, v in best_run[best_name]["params"].items():
            print(f"    {k} = {v}")

    # ─── Feature importances (first run) ───────────────────────────────────
    print("\n=== Top features (tree models, run 1) ===")
    for m, res in all_runs[0].items():
        if res["imp"]:
            print("  " + m + ": " + ", ".join(f"{f} ({v:.3f})" for f, v in res["imp"]))
            if res["params"]:
                print("    params: " + str(res["params"]))

    # ─── CI plot ───────────────────────────────────────────────────────────
    acc_mu = np.array([float(r.split()[0]) for r in detailed["Test-Acc"]])
    acc_sd = np.array([float(r.split("±")[1]) for r in detailed["Test-Acc"]])
    plt.errorbar(np.arange(len(acc_mu)), acc_mu, yerr=2*acc_sd, fmt="o")
    plt.xticks(np.arange(len(acc_mu)), detailed["Model"])
    plt.ylabel("Accuracy")
    plt.title("Test accuracy ± 2 σ")
    plt.ylim(0.7, 1.01)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return detailed

# ──────────────────────────────────────────────────────────────────────────────
# 6.  CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="data/final_data_with_patterns.csv")
    ap.add_argument("--k",    type=int, default=15, help="top-k features")
    ap.add_argument("--runs", type=int, default=5,  help="random seeds")
    ap.add_argument("--tune", action="store_true",  help="enable GridSearch")
    args = ap.parse_args()

    run_pipeline(csv=args.csv, k_best=args.k, runs=args.runs, tune=args.tune)
