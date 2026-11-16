"""The fairness_weight sweep behind the choice in run_experiment.py.

One LFR fit per dataset per weight, plus the logistic baseline for reference,
all on the same split as run_experiment.py. Appends one row per fit to
results/sweep.csv, so it is safe to interrupt and restart (one weight per
process is fine, they are independent).

  python src/sweep.py
"""

import os
import sys
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.preprocessing import PrototypeRepresentationLearner

import data_loading
from run_experiment import (DATASETS, SEED, N_PROTOTYPES, MAX_ITER, WEIGHTS,
                            measure, split)

OUT = "results/sweep.csv"


def sweep_one(name, X, y, s):
    Xtr, Xte, ytr, yte, s_tr, s_te = split(X, y, s)
    s_tr, s_te = s_tr.values, s_te.values

    rows = []
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    rows.append({"dataset": name, "fairness_weight": "baseline-lr",
                 **measure(Xte, yte, s_te, lr.predict(Xte))})

    for w in WEIGHTS:
        t0 = time.time()
        prl = PrototypeRepresentationLearner(n_prototypes=N_PROTOTYPES,
                                             fairness_weight=w,
                                             max_iter=MAX_ITER,
                                             random_state=SEED)
        prl.fit(Xtr, ytr, sensitive_features=s_tr)
        rows.append({"dataset": name, "fairness_weight": w,
                     "iterations": int(prl.n_iter_),
                     "seconds": round(time.time() - t0, 1),
                     **measure(Xte, yte, s_te, prl.predict(Xte))})
        print(f"[{name}] w={w}: {time.time() - t0:.0f}s, "
              f"{prl.n_iter_} iterations")
        sys.stdout.flush()
    return rows


if __name__ == "__main__":
    name = sys.argv[1]
    X, y, s = DATASETS[name]()
    new = pd.DataFrame(sweep_one(name, X, y, s))

    old = pd.read_csv(OUT) if os.path.exists(OUT) else None
    if old is not None:
        old = old[old["dataset"] != name]
        pd.concat([old, new], ignore_index=True).to_csv(OUT, index=False)
    else:
        new.to_csv(OUT, index=False)
    print(new.to_string(index=False))
