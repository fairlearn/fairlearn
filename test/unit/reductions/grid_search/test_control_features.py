# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, ErrorRateParity,\
    TruePositiveRateParity, FalsePositiveRateParity

# Set up a loan scenario, with three income bands A, B & C and
# one sensitive attribute with values F & G

ibs = ["A", "B", "C"]
sfs = ["F", "G"]

# Numbers for each intersection
n = {
    "A": {"F": 80, "G": 120},
    "B": {"F": 300, "G": 350},
    "C": {"F": 800, "G": 650},
}

# Approval rates for each intersection
f = {
    "A": {"F": 0.5, "G": 0.9},
    "B": {"F": 0.4, "G": 0.7},
    "C": {"F": 0.1, "G": 0.5},
}


def _generate_data():
    IB = []
    SF = []
    PLOAN = []
    Y = []

    for ib in ibs:
        for sf in sfs:
            n_curr = n[ib][sf]
            f_curr = f[ib][sf]

            for i in range(n_curr):
                IB.append(ib)
                SF.append(sf)
                flip = random.random()
                PLOAN.append(flip)
                Y.append(1 if flip < f_curr else 0)

    X = pd.DataFrame(data=np.transpose([IB, SF, PLOAN]), columns=["ctrl", "sens", "ploan"])

    return X, Y


def generate_disparities(grid_search, X, y, sf, cf):
    result = []
    for p in grid_search.predictors_:
        y_pred = p.predict(X)

        mf = MetricFrame(selection_rate,
                         y, y_pred,
                         sensitive_features=sf,
                         control_features=cf)

        result.append(mf.difference(method='to_overall'))
    return result


def run_comparisons(moment, metric_fn):
    num_grid_points=51
    X, y = _generate_data()
    X_dummy = pd.get_dummies(X)

    mf_input = MetricFrame(metric_fn, y, y,
                           sensitive_features=X['sens'],
                           control_features=X['ctrl'])

    print("Metric for input:\n", mf_input.by_group)
    print("Input Metric differences:\n", mf_input.difference(method='to_overall'), "\n")

    unmitigated = LogisticRegression()
    unmitigated.fit(X_dummy, y)
    y_pred = unmitigated.predict(X_dummy)
    mf_unmitigated = MetricFrame(metric_fn,
                                 y, y_pred,
                                 sensitive_features=X['sens'],
                                 control_features=X['ctrl'])
    print("Unmitigated metric:\n", mf_unmitigated.by_group)
    print("Unmitigated metric differences:\n",
          mf_unmitigated.difference(method='to_overall'), "\n")

    gs_basic = GridSearch(
        LogisticRegression(),
        constraints=moment(),
        grid_size=num_grid_points)
    gs_basic.fit(X_dummy, y, sensitive_features=X['sens'])
    y_pred_basic = gs_basic.predict(X_dummy)
    mf_basic = MetricFrame(metric_fn, y, y_pred_basic,
                           sensitive_features=X['sens'],
                           control_features=X['ctrl'])
    print("Basic expgrad metric:\n", mf_basic.by_group)
    print("Basic expgrad metric differences:\n",
          mf_basic.difference(method='to_overall'), "\n")

    gs_control = GridSearch(
        LogisticRegression(),
        constraints=moment(),
        grid_size=num_grid_points,
        grid_limit=0.5)
    gs_control.fit(X_dummy, y,
                   sensitive_features=X['sens'],
                   control_features=X['ctrl'])
    y_pred_control = gs_control.predict(X_dummy)
    mf_control = MetricFrame(metric_fn, y, y_pred_control,
                             sensitive_features=X['sens'],
                             control_features=X['ctrl'])
    print("expgrad_control metric:\n", mf_control.by_group)
    print("expgrad_control metric differences:\n",
          mf_control.difference(method='to_overall'))

    diffs_control = generate_disparities(gs_control, X_dummy, y, X['sens'], X['ctrl'])

    plt.plot([y["A"] for y in diffs_control], label='A')
    plt.plot([y["B"] for y in diffs_control], label='B')
    plt.plot([y["C"] for y in diffs_control], label='C')
    plt.legend()
    plt.show()

    assert (mf_control.difference(method='to_overall') <=
            mf_unmitigated.difference(method='to_overall')).all()

    assert (mf_control.difference(method='to_overall') <=
            mf_basic.difference(method='to_overall')).all()


def test_demographic_parity():
    run_comparisons(DemographicParity, selection_rate)
