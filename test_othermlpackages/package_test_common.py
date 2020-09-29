# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Common testing methods for use with other ML packages."""

import copy

import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.metrics import demographic_parity_difference
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

    # Ensure indices are aligned
    X_train = X_train.reset_index(drop=True)
    A_train = A_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    A_test = A_test.reset_index(drop=True)

    return X_train, Y_train, A_train, X_test, Y_test, A_test


def run_expgrad_classification(estimator, moment):
    """Run classification test with ExponentiatedGradient."""
    X_train, Y_train, A_train, X_test, Y_test, A_test = fetch_adult()
    verification_moment = copy.deepcopy(moment)

    unmitigated = copy.deepcopy(estimator)
    unmitigated.fit(X_train, Y_train)

    expgrad = ExponentiatedGradient(
        estimator,
        constraints=moment)
    expgrad.fit(X_train, Y_train, sensitive_features=A_train)

    assert expgrad.n_oracle_calls_ > 1
    assert len(expgrad.predictors_) > 1

    verification_moment.load_data(X_test, Y_test, sensitive_features=A_test)
    gamma_unmitigated = verification_moment.gamma(lambda x: unmitigated.predict(x))
    gamma_mitigated = verification_moment.gamma(lambda x: expgrad.predict(x))

    for idx in gamma_mitigated.index:
        assert abs(gamma_mitigated[idx]) <= abs(gamma_unmitigated[idx]), "Checking {0}".format(idx)


def run_gridsearch_classification(estimator, moment):
    """Run classification test with GridSearch."""
    X_train, Y_train, A_train, X_test, Y_test, A_test = fetch_adult()
    verification_moment = copy.deepcopy(moment)

    unmitigated = copy.deepcopy(estimator)
    unmitigated.fit(X_train, Y_train)

    num_predictors = 11
    gs = GridSearch(
        estimator,
        constraints=moment,
        grid_size=num_predictors)
    gs.fit(X_train, Y_train, sensitive_features=A_train)

    assert len(gs.predictors_) == num_predictors

    verification_moment.load_data(X_test, Y_test, sensitive_features=A_test)
    gamma_unmitigated = verification_moment.gamma(lambda x: unmitigated.predict(x))
    gamma_mitigated = verification_moment.gamma(lambda x: gs.predict(x))

    for idx in gamma_mitigated.index:
        assert abs(gamma_mitigated[idx]) <= abs(gamma_unmitigated[idx]), "Checking {0}".format(idx)


def run_thresholdoptimizer_classification(estimator):
    """Run classification test with ThresholdOptimizer."""
    X_train, Y_train, A_train, X_test, Y_test, A_test = fetch_adult()

    unmitigated = copy.deepcopy(estimator)
    unmitigated.fit(X_train, Y_train)
    unmitigated_predictions = unmitigated.predict(X_test)

    to = ThresholdOptimizer(estimator=estimator, prefit=False)
    to.fit(X_train, Y_train, sensitive_features=A_train)

    mitigated_predictions = to.predict(X_test, sensitive_features=A_test)

    dp_diff_unmitigated = demographic_parity_difference(Y_test,
                                                        unmitigated_predictions,
                                                        sensitive_features=A_test)

    dp_diff_mitigated = demographic_parity_difference(Y_test,
                                                      mitigated_predictions,
                                                      sensitive_features=A_test)
    assert dp_diff_mitigated <= dp_diff_unmitigated
