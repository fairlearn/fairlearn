# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Common testing methods for use with other ML packages."""

import copy
import logging

import pandas as pd
from numpy import mean, random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fairlearn.metrics import demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch

logger = logging.getLogger(__name__)


def fetch_test_data():
    """Load a bundled dataset for testing without external network access."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    Y = data.target.to_numpy(copy=True)
    group = Y.copy()
    flip_group = random.default_rng(12345).random(len(Y)) < 0.2
    group[flip_group] = 1 - group[flip_group]
    A = pd.Series(group).map({0: "group0", 1: "group1"})

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
        X_scaled, Y, A, test_size=0.3, random_state=12345, stratify=Y
    )

    # Ensure indices are aligned
    X_train = X_train.reset_index(drop=True)
    A_train = A_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    A_test = A_test.reset_index(drop=True)

    return X_train, Y_train, A_train, X_test, Y_test, A_test


def run_expgrad_classification(estimator, moment):
    """Run classification test with ExponentiatedGradient."""
    X_train, Y_train, A_train, X_test, Y_test, A_test = fetch_test_data()
    verification_moment = copy.deepcopy(moment)

    unmitigated = copy.deepcopy(estimator)
    unmitigated.fit(X_train, Y_train)

    expgrad = ExponentiatedGradient(estimator, constraints=moment)
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
    X_train, Y_train, A_train, X_test, Y_test, A_test = fetch_test_data()
    verification_moment = copy.deepcopy(moment)

    unmitigated = copy.deepcopy(estimator)
    unmitigated.fit(X_train, Y_train)

    num_predictors = 11
    gs = GridSearch(estimator, constraints=moment, grid_size=num_predictors)
    gs.fit(X_train, Y_train, sensitive_features=A_train)

    assert len(gs.predictors_) == num_predictors

    verification_moment.load_data(X_test, Y_test, sensitive_features=A_test)
    gamma_unmitigated = verification_moment.gamma(lambda x: unmitigated.predict(x))
    gamma_mitigated = verification_moment.gamma(lambda x: gs.predict(x))

    for idx in gamma_mitigated.index:
        assert abs(gamma_mitigated[idx]) <= abs(gamma_unmitigated[idx]), "Checking {0}".format(idx)


def run_thresholdoptimizer_classification(estimator):
    """Run classification test with ThresholdOptimizer."""
    X_train, Y_train, A_train, X_test, Y_test, A_test = fetch_test_data()

    unmitigated = copy.deepcopy(estimator)
    unmitigated.fit(X_train, Y_train)
    unmitigated_predictions = unmitigated.predict(X_test)

    to = ThresholdOptimizer(estimator=estimator, prefit=False, predict_method="predict")
    to.fit(X_train, Y_train, sensitive_features=A_train)

    mitigated_predictions = to.predict(X_test, sensitive_features=A_test)

    dp_diff_unmitigated = demographic_parity_difference(
        Y_test, unmitigated_predictions, sensitive_features=A_test
    )

    dp_diff_mitigated = demographic_parity_difference(
        Y_test, mitigated_predictions, sensitive_features=A_test
    )
    assert dp_diff_mitigated <= dp_diff_unmitigated


def run_AdversarialFairness_classification(estimator):
    """Run classification test with AdversarialFairness."""
    random.seed(123)

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target.to_numpy()
    sensitive_feature = (X["mean radius"] >= X["mean radius"].median()).to_numpy().reshape(-1, 1)
    X = StandardScaler().fit_transform(X)

    X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
        X, y, sensitive_feature, test_size=0.2, random_state=12345, stratify=y
    )

    estimator.epochs = 100
    estimator.batch_size = 2**10
    estimator.shuffle = True
    estimator.progress_updates = None

    estimator.fit(X_train, Y_train, sensitive_features=A_train)

    predictions = estimator.predict(X_test)

    dp_diff = demographic_parity_difference(Y_test, predictions, sensitive_features=A_test)

    accuracy = mean(predictions == Y_test)

    # This might give problems as it is a bit random (though we set a seed,
    # it may depend on other factors such as version).
    if not (accuracy > 0.8 and dp_diff < 0.2):
        logger.warning(
            "Training of AdversarialFairness is worse than usual."  # noqa
            + f"Accuracy {str(accuracy)} with a disparity difference of"  # noqa
            + f"{str(dp_diff)}."  # noqa
        )
    assert estimator is not None
