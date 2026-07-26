# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from test.unit.reductions.data_generators import loan_scenario_generator

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.reductions import (
    DemographicParity,
    EqualizedOdds,
    ErrorRateParity,
    FalsePositiveRateParity,
    TruePositiveRateParity,
)

# Set up a loan scenario, with three income bands A, B & C and
# one sensitive attribute with values F & G

ibs = ["A", "B", "C"]
sfs = ["F", "G"]

# Numbers for each intersection
n = {
    "A": {"F": 90, "G": 140},
    "B": {"F": 300, "G": 350},
    "C": {"F": 900, "G": 750},
}

# Approval rates for each intersection
f = {
    "A": {"F": 0.6, "G": 0.9},
    "B": {"F": 0.4, "G": 0.7},
    "C": {"F": 0.2, "G": 0.5},
}


def _simple_compare(moment, metric):
    X, y = loan_scenario_generator(n, f, sfs, ibs, seed=7632752)
    X_dummy = pd.get_dummies(X)

    est = LogisticRegression()
    est.fit(X_dummy, y)
    y_pred = est.predict(X_dummy)

    target = moment()
    target.load_data(
        X_dummy,
        y,
        sensitive_features=X["sens"],
        control_features=X["ctrl"],
    )

    # gamma measures the constraint violation relative to the overall value
    results = target.gamma(est.predict)

    # Compute the constraint violation using the metrics
    mf_pred = MetricFrame(
        metrics=metric,
        y_true=y,
        y_pred=y_pred,
        sensitive_features=X["sens"],
        control_features=X["ctrl"],
    )
    diffs = mf_pred.by_group - mf_pred.overall

    # Compare (with a very small amount of wriggle room)
    for ib in ibs:
        for sf in sfs:
            event_format = "control={0},all"
            assert diffs[(ib, sf)] == pytest.approx(
                results[("+", event_format.format(ib), sf)], rel=1e-10, abs=1e-12
            )
            assert diffs[(ib, sf)] == pytest.approx(
                -results[("-", event_format.format(ib), sf)], rel=1e-10, abs=1e-12
            )


def test_demographic_parity():
    _simple_compare(DemographicParity, selection_rate)


def test_error_rate_parity():
    def err_rate(y_true, y_pred):
        return 1 - accuracy_score(y_true, y_pred)

    _simple_compare(ErrorRateParity, err_rate)


def _selected_label_compare(moment, metric, selected_label):
    # Similar to _simple_compare, but we need to worry about the y label
    X, y = loan_scenario_generator(n, f, sfs, ibs, seed=7132752)
    X_dummy = pd.get_dummies(X)

    est = LogisticRegression()
    est.fit(X_dummy, y)
    y_pred = est.predict(X_dummy)

    target = moment()
    target.load_data(
        X_dummy,
        y,
        sensitive_features=X["sens"],
        control_features=X["ctrl"],
    )

    # gamma measures the constraint violation relative to the overall value
    results = target.gamma(est.predict)

    # Compute the constraint violation using the metrics
    mf_pred = MetricFrame(
        metrics=metric,
        y_true=y,
        y_pred=y_pred,
        sensitive_features=X["sens"],
        control_features=X["ctrl"],
    )
    diffs = mf_pred.by_group - mf_pred.overall

    # Compare (with a very small amount of wriggle room)
    for ib in ibs:
        for sf in sfs:
            # Format defined within utility_parity._combine_event_and_control
            label_format = "control={0},label={1}"
            label = label_format.format(ib, selected_label)
            assert diffs[(ib, sf)] == pytest.approx(
                results[("+", label, sf)], rel=1e-10, abs=1e-12
            )
            assert diffs[(ib, sf)] == pytest.approx(
                -results[("-", label, sf)], rel=1e-10, abs=1e-12
            )


def test_true_positive_parity():
    moment = TruePositiveRateParity
    metric = true_positive_rate
    selected_label = 1

    _selected_label_compare(moment, metric, selected_label)


def test_false_positive_parity():
    moment = FalsePositiveRateParity
    metric = false_positive_rate
    selected_label = 0

    _selected_label_compare(moment, metric, selected_label)


def test_equalized_odds():
    moment = EqualizedOdds

    metric = true_positive_rate
    selected_label = 1
    _selected_label_compare(moment, metric, selected_label)

    metric = false_positive_rate
    selected_label = 0
    _selected_label_compare(moment, metric, selected_label)
