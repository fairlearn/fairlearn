"""The experiment behind the tables in the README.

Three settings on the same held-out split, per dataset:
  labels_before  the fairness measures of the raw labels, before any training
  baseline       logistic regression on the features, no mitigation
  lfr            prototype representation fitted on the train fold, logistic
                 head on the prototype coordinates, predicting the test fold

  python src/run_experiment.py     (from the repo root)
"""

import json
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.preprocessing import PrototypeRepresentationLearner

import data_loading
import metrics

SEED = 42
N_PROTOTYPES = 10
MAX_ITER = 100
DATASETS = {
    "adult": data_loading.load_adult,
    "german": data_loading.load_german,
    "compas": data_loading.load_compas,
}
# from the sweep in results/sweep.csv, see the README
FAIRNESS_WEIGHT = {"adult": 50.0, "german": 5.0, "compas": 20.0}
WEIGHTS = [0.0, 1.0, 5.0, 20.0, 50.0]


def split(X, y, s):
    Xe = pd.get_dummies(X, drop_first=True).astype(float)
    # stratify on label and group so both keep their share in the test fold
    strat = y.astype(str) + "_" + s.astype(str)
    Xtr, Xte, ytr, yte, s_tr, s_te = train_test_split(
        Xe, y, s, test_size=0.2, random_state=SEED, stratify=strat)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte, s_tr, s_te


def measure(Xte, yte, s_te, pred):
    return {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "dp_diff": round(float(metrics.dp_diff(pred, s_te)), 4),
        "dir": round(float(metrics.dir_ratio(pred, s_te)), 3),
        "consistency": round(float(metrics.consistency(Xte, pred)), 4),
    }


def run_one(name, X, y, s):
    Xtr, Xte, ytr, yte, s_tr, s_te = split(X, y, s)
    s_tr, s_te = s_tr.values, s_te.values

    out = {
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xte.shape[1]),
        "base_rate_unpriv": round(float(yte[s_te == 1].mean()), 4),
        "base_rate_priv": round(float(yte[s_te == 0].mean()), 4),
    }

    # the raw labels, scored against the test fold's own labels
    raw = yte.values
    out["labels_before"] = measure(Xte, yte, s_te, raw)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr, ytr)
    out["baseline_lr"] = measure(Xte, yte, s_te, lr.predict(Xte))

    t0 = time.time()
    prl = PrototypeRepresentationLearner(n_prototypes=N_PROTOTYPES,
                                         fairness_weight=FAIRNESS_WEIGHT[name],
                                         max_iter=MAX_ITER,
                                         random_state=SEED)
    prl.fit(Xtr, ytr, sensitive_features=s_tr)
    out["lfr"] = measure(Xte, yte, s_te, prl.predict(Xte))
    print(f"[{name}] lfr fitted in {time.time() - t0:.0f}s, "
          f"{prl.n_iter_} iterations")
    return out


def main():
    results = {}
    for name, load in DATASETS.items():
        X, y, s = load()
        results[name] = run_one(name, X, y, s)
        r = results[name]
        print(f"[{name}] dp_diff {r['labels_before']['dp_diff']:+.4f} -> "
              f"{r['lfr']['dp_diff']:+.4f}, accuracy {r['lfr']['accuracy']:.4f}, "
              f"f1 {r['lfr']['f1']:.4f}")

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/results.json")


if __name__ == "__main__":
    main()
