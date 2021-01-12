# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity, ErrorRateParity,\
    TruePositiveRateParity, FalsePositiveRateParity, EqualizedOdds

from test.unit.reductions.data_generators import loan_scenario_generator

# Set up a loan scenario, with three income bands A, B & C and
# one sensitive attribute with values F & G

ibs = ["A", "B", "C"]
sfs = ["F", "G"]

# Numbers for each intersection
n = {
    "A": {"F": 80, "G": 100},
    "B": {"F": 200, "G": 50},
    "C": {"F": 400, "G": 350},
}

# Approval rates for each intersection
f = {
    "A": {"F": 0.5, "G": 0.9},
    "B": {"F": 0.4, "G": 0.7},
    "C": {"F": 0.1, "G": 0.5},
}


def run_comparisons(moment, metric_fn):
    X, y = loan_scenario_generator(n, f, sfs, ibs, seed=163)
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

    expgrad_basic = ExponentiatedGradient(
        LogisticRegression(),
        constraints=moment(),
        eps=0.005)
    expgrad_basic.fit(X_dummy, y, sensitive_features=X['sens'])
    y_pred_basic = expgrad_basic.predict(X_dummy, random_state=8235)
    mf_basic = MetricFrame(metric_fn, y, y_pred_basic,
                           sensitive_features=X['sens'],
                           control_features=X['ctrl'])
    print("Basic expgrad metric:\n", mf_basic.by_group)
    print("Basic expgrad metric differences:\n",
          mf_basic.difference(method='to_overall'), "\n")

    expgrad_control = ExponentiatedGradient(
        LogisticRegression(),
        constraints=moment(),
        eps=0.005)
    expgrad_control.fit(X_dummy, y,
                        sensitive_features=X['sens'],
                        control_features=X['ctrl'])
    y_pred_control = expgrad_control.predict(X_dummy, random_state=852)
    mf_control = MetricFrame(metric_fn, y, y_pred_control,
                             sensitive_features=X['sens'],
                             control_features=X['ctrl'])
    print("expgrad_control metric:\n", mf_control.by_group)
    print("expgrad_control metric differences:\n",
          mf_control.difference(method='to_overall'))

    assert (mf_control.difference(method='to_overall') <=
            mf_unmitigated.difference(method='to_overall')).all()

    assert (mf_control.difference(method='to_overall') <=
            mf_basic.difference(method='to_overall')).all()


def test_demographic_parity():
    run_comparisons(DemographicParity, selection_rate)


def test_error_rate_parity():
    run_comparisons(ErrorRateParity, accuracy_score)


def test_true_positive_rate_parity():
    run_comparisons(TruePositiveRateParity, true_positive_rate)


def test_false_positive_rate_parity():
    run_comparisons(FalsePositiveRateParity, false_positive_rate)


def test_equalized_odds():
    # Have to do this one longhand, since it combines tpr and fpr
    X, y = loan_scenario_generator(n, f, sfs, ibs, seed=632753)
    X_dummy = pd.get_dummies(X)

    metrics = {'tpr': true_positive_rate, 'fpr': false_positive_rate}

    unmitigated = LogisticRegression()
    unmitigated.fit(X_dummy, y)
    y_pred = unmitigated.predict(X_dummy)
    mf_unmitigated = MetricFrame(metrics,
                                 y, y_pred,
                                 sensitive_features=X['sens'],
                                 control_features=X['ctrl'])

    expgrad_basic = ExponentiatedGradient(
        LogisticRegression(),
        constraints=EqualizedOdds(difference_bound=0.01),
        eps=0.01)
    expgrad_basic.fit(X_dummy, y, sensitive_features=X['sens'])
    y_pred_basic = expgrad_basic.predict(X_dummy, random_state=9235)
    mf_basic = MetricFrame(metrics, y, y_pred_basic,
                           sensitive_features=X['sens'],
                           control_features=X['ctrl'])

    expgrad_control = ExponentiatedGradient(
        LogisticRegression(),
        constraints=EqualizedOdds(difference_bound=0.01),
        eps=0.01)
    expgrad_control.fit(X_dummy, y,
                        sensitive_features=X['sens'],
                        control_features=X['ctrl'])
    y_pred_control = expgrad_control.predict(X_dummy, random_state=8152)
    mf_control = MetricFrame(metrics, y, y_pred_control,
                             sensitive_features=X['sens'],
                             control_features=X['ctrl'])

    compare_unmitigated = (mf_control.difference(method='to_overall') <=
                           mf_unmitigated.difference(method='to_overall'))
    print(compare_unmitigated)

    compare_basic = (mf_control.difference(method='to_overall') <=
                     mf_basic.difference(method='to_overall'))
    print(compare_basic)

    assert compare_basic.values.reshape(6).all()
    assert compare_unmitigated.values.reshape(6).all()
