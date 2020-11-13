# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression

from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

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


def test_demographic_parity_controlfeatures():
    X, y = _generate_data()
    X_dummy = pd.get_dummies(X)

    mf_input = MetricFrame(selection_rate, y, y,
                           sensitive_features=X['sens'],
                           control_features=X['ctrl'])

    print(mf_input.overall)
    print(mf_input.by_group)
    print(mf_input.difference(method='to_overall'))

    unmitigated = LogisticRegression()
    unmitigated.fit(X_dummy, y)
    y_pred = unmitigated.predict(X_dummy)
    mf_unmitigated = MetricFrame(selection_rate,
                                 y, y_pred,
                                 sensitive_features=X['sens'],
                                 control_features=X['ctrl'])
    print(mf_unmitigated.overall)
    print(mf_unmitigated.by_group)

    expgrad_basic = ExponentiatedGradient(
        LogisticRegression(),
        constraints=DemographicParity(),
        eps=0.01)
    expgrad_basic.fit(X_dummy, y, sensitive_features=X['sens'])
    y_pred_basic = expgrad_basic.predict(X_dummy)
    mf_basic = MetricFrame(selection_rate, y, y_pred_basic,
                           sensitive_features=X['sens'],
                           control_features=X['ctrl'])
    print(mf_basic.by_group)
    print(mf_basic.difference(method='to_overall'))

    expgrad_control = ExponentiatedGradient(
        LogisticRegression(),
        constraints=DemographicParity(),
        eps=0.01)
    expgrad_control.fit(X_dummy, y,
                        sensitive_features=X['sens'],
                        control_features=X['ctrl'])
    y_pred_control = expgrad_control.predict(X_dummy)
    mf_control = MetricFrame(selection_rate, y, y_pred_control,
                             sensitive_features=X['sens'],
                             control_features=X['ctrl'])
    print(mf_control.by_group)
    print(mf_control.difference(method='to_overall'))
