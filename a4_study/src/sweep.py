"""Validation sweep behind the hyper-parameter choice in run_experiment.py.

For each dataset, one fairness weight per row of the grid; accuracy, F1 and
the three fairness measures are logged per setting. Writes results/sweep.csv.

Run from the repository root:
    python src/sweep.py
"""

import sys
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fairlearn.preprocessing import PrototypeRepresentationLearner

from data_loading import load_adult, load_german, load_compas
from metrics import demographic_parity_diff, disparate_impact_ratio, consistency

SEED = 42
N_PROTOTYPES = 10
WEIGHTS = [0.0, 1.0, 5.0, 20.0, 50.0]
# 100 is plenty: every fit in results/sweep.csv converged in 24-95 iterations
MAX_ITER = 100


def sweep(name, X, y, s):
    Xe = pd.get_dummies(X, drop_first=True).astype(float)
    strat = y.astype(str) + "_" + s.astype(str)
    Xtr, Xte, ytr, yte, s_tr, s_te = train_test_split(
        Xe, y, s, test_size=0.2, random_state=SEED, stratify=strat)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    rows = []

    def add(w, pred):
        rows.append({
            "dataset": name, "fairness_weight": w,
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "f1": round(float(f1_score(yte, pred)), 4),
            "dp_diff": round(float(demographic_parity_diff(pred, s_te.values)), 4),
            "dir": round(float(disparate_impact_ratio(pred, s_te.values)), 3),
            "consistency": round(float(consistency(Xte, pred)), 4),
        })

    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    add("baseline-lr", lr.predict(Xte))

    for w in WEIGHTS:
        t0 = time.time()
        prl = PrototypeRepresentationLearner(n_prototypes=N_PROTOTYPES,
                                             fairness_weight=w,
                                             max_iter=MAX_ITER,
                                             random_state=SEED)
        prl.fit(Xtr, ytr, sensitive_features=s_tr)
        add(w, prl.predict(Xte))
        print(f"[{name}] w={w}: {time.time() - t0:.0f}s")
        sys.stdout.flush()

    return rows


if __name__ == "__main__":
    all_rows = []
    for name, loader in [("german", load_german), ("compas", load_compas),
                         ("adult", load_adult)]:
        X, y, s = loader()
        all_rows.extend(sweep(name, X, y, s))
    df = pd.DataFrame(all_rows)
    df.to_csv("results/sweep.csv", index=False)
    print(df.to_string(index=False))
