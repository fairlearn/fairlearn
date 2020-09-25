# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Common testing methods for use with other ML packages."""

import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch


def fetch_adult():
    """Grab dataset for testing."""
    data = fetch_openml(data_id=1590, as_frame=True)
    X = data.data.drop(labels=['sex'], axis=1)
    X = pd.get_dummies(X)
    Y = (data.target == '>50K') * 1
    A = data.data['sex']

    le = LabelEncoder()
    Y = le.fit_transform(Y)

    le = LabelEncoder()

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, Y_train, Y_test, A_train, A_test = \
        train_test_split(X_scaled, Y, A,
                         test_size=0.3,
                         random_state=12345,
                         stratify=Y)

    return X_train, Y_train, A_train


def run_expgrad_classification(estimator, moment):
    """Run classification test with ExponentiatedGradient."""
    X, Y, A = fetch_adult()

    expgrad = ExponentiatedGradient(
        estimator,
        constraints=moment)
    expgrad.fit(X, Y, sensitive_features=A)

    assert expgrad.n_oracle_calls_ > 1
    assert len(expgrad.predictors_) > 1


def run_gridsearch_classification(estimator, moment):
    """Run classification test with GridSearch."""
    X, Y, A = fetch_adult()

    num_predictors = 5
    gs = GridSearch(
        estimator,
        constraints=moment,
        grid_size=num_predictors)
    gs.fit(X, Y, sensitive_features=A)

    assert len(gs.predictors_) == num_predictors


def run_thresholdoptimizer_classification(estimator):
    """Run classification test with ThresholdOptimizer."""
    X, Y, A = fetch_adult()

    to = ThresholdOptimizer(estimator=estimator, prefit=False)
    to.fit(X, Y, sensitive_features=A)

    results = to.predict(X, sensitive_features=A)
    assert results is not None
