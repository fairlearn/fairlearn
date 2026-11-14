"""Experiment: LFR (Zemel et al. 2013) with fairlearn's PrototypeRepresentationLearner.

For every dataset we report, on the same held-out test split:
  - labels-before   : DP_Diff, DIR and consistency of the original labels,
                      i.e. before training any classifier
  - baseline        : plain logistic regression on the features, no mitigation
  - LFR             : prototype representation fitted on the training data,
                      logistic head on top, predictions on the test features

Run from the repository root:
    python src/run_experiment.py
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fairlearn.preprocessing import PrototypeRepresentationLearner

from data_loading import load_adult, load_german, load_compas
from metrics import demographic_parity_diff, disparate_impact_ratio, consistency

SEED = 42
# picked on the validation sweep in results/sweep.csv, see README
N_PROTOTYPES = 10
MAX_ITER = 100  # all sweep fits converged within 24-95 iterations at tol 1e-6
FAIRNESS_WEIGHTS = {"adult": 50.0, "german": 5.0, "compas": 20.0}

DATASETS = {
    "adult": load_adult,
    "german": load_german,
    "compas": load_compas,
}


def run_one(name, X, y, s):
    Xe = pd.get_dummies(X, drop_first=True).astype(float)

    # stratify on label AND group so both keep their share in the test split
    strat = y.astype(str) + "_" + s.astype(str)
    Xtr, Xte, ytr, yte, s_tr, s_te = train_test_split(
        Xe, y, s, test_size=0.2, random_state=SEED, stratify=strat)
    s_tr, s_te = s_tr.values, s_te.values

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    rows = {}
    rows["n_train"] = int(Xtr.shape[0])
    rows["n_test"] = int(Xte.shape[0])
    rows["n_features"] = int(Xe.shape[1])
    rows["base_rate_unpriv"] = round(float(yte[s_te == 1].mean()), 4)
    rows["base_rate_priv"] = round(float(yte[s_te == 0].mean()), 4)

    def measure(tag, pred, scored=True):
        rows[tag] = {
            "dp_diff": round(float(demographic_parity_diff(pred, s_te)), 4),
            "dir": round(float(disparate_impact_ratio(pred, s_te)), 3),
            "consistency": round(float(consistency(Xte, pred)), 4),
        }
        if scored:
            rows[tag]["accuracy"] = round(float(accuracy_score(yte, pred)), 4)
            rows[tag]["f1"] = round(float(f1_score(yte, pred)), 4)

    # no-mitigation reference points
    measure("labels_before", yte.values, scored=False)

    t0 = time.time()
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    measure("baseline_lr", lr.predict(Xte))

    # the mitigation itself: fit the representation on the training fold only
    prl = PrototypeRepresentationLearner(
        n_prototypes=N_PROTOTYPES,
        fairness_weight=FAIRNESS_WEIGHTS[name],
        max_iter=MAX_ITER,
        random_state=SEED,
    )
    prl.fit(Xtr, ytr.values, sensitive_features=s_tr)
    measure("lfr", prl.predict(Xte))
    print(f"[{name}] done in {time.time() - t0:.0f}s (LFR iterations: {prl.n_iter_})")
    return rows


def main():
    results = {}
    for name, loader in DATASETS.items():
        X, y, s = loader()
        results[name] = run_one(name, X, y, s)
        print(f"[{name}] dp_diff {results[name]['labels_before']['dp_diff']:+.4f}"
              f" -> {results[name]['lfr']['dp_diff']:+.4f}, "
              f"accuracy {results[name]['lfr']['accuracy']:.4f}")

    out = "results/results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
