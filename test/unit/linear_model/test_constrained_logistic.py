# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

from fairlearn.linear_model._constrained_logistic import (
    ConstrainedLogisticRegression,
)


def test_unconstrained_vs_normal_lr(data_X, data_y, constraints=None):
    normal_lr = LogisticRegression(random_state=0).fit(data_X, data_y)
    unconstrained = ConstrainedLogisticRegression(
        random_state=0, constraints=constraints
    ).fit(data_X, data_y)

    assert normal_lr.score(data_X, data_y) == unconstrained.score(data_X, data_y)
    assert np.all(
        normal_lr.predict_proba(data_X) == unconstrained.predict_proba(data_X)
    )


def test_one_sensitive_feature(data_X, data_y, data_single_sf):
    n_samples = len(data_y)
    classes = np.unique(data_y)
    n_classes = classes.shape[0]

    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0
    )
    predicted = clf.fit(data_X, data_y, sensitive_features=data_single_sf).predict(
        data_X
    )
    assert predicted.shape == (n_samples,)

    probabilities = clf.predict_proba(data_X)
    assert probabilities.shape == (n_samples, n_classes)


def test_two_sensitive_features(data_X, data_y, data_multiple_sf):
    n_samples = len(data_y)
    classes = np.unique(data_y)
    n_classes = classes.shape[0]

    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0
    )
    predicted = clf.fit(data_X, data_y, sensitive_features=data_multiple_sf).predict(
        data_X
    )
    assert predicted.shape == (n_samples,)

    probabilities = clf.predict_proba(data_X)
    assert probabilities.shape == (n_samples, n_classes)


def test_multinomial_classification():
    # Abalone dataset for multiple classes
    data = fetch_openml(data_id=183, as_frame=True)
    X = data.data
    X = X.drop("Sex", axis=1)
    sensitive_features = data.data[["Sex"]]
    y = data.target
    y = y.to_numpy()
    n_samples = len(y)
    n_classes = np.unique(y).shape[0]
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0, n_jobs=-1
    )
    predicted = clf.fit(X, y, sensitive_features=sensitive_features).predict(X)
    assert predicted.shape == (n_samples,)
    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (n_samples, n_classes)


def test_too_many_cov_bound_values(data_X, data_y, data_multiple_sf):
    num_sens_features = data_multiple_sf.shape[0]
    covariance_bound = [0] * (num_sens_features + 1)
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=covariance_bound, n_jobs=-1
    )
    with pytest.raises(
        ValueError, match="^Number of covariance bound values can not exceed"
    ):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_wrong_solver(data_X, data_y, data_multiple_sf):
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", solver="lbfgs", covariance_bound=0, n_jobs=-1
    )
    with pytest.raises(
        ValueError, match="^Constrained Logistic Regression supports only solvers in"
    ):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_mismatch_X_sf_rows(data_X, data_y, data_multiple_sf):
    data_multiple_sf = data_multiple_sf[:-1, :]
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0, n_jobs=-1
    )
    with pytest.raises(ValueError, match="^X has [0-9]+ instances while"):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_sf_wrong_type(data_X, data_y, data_multiple_sf):
    data_multiple_sf = tuple(data_multiple_sf)
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0, n_jobs=-1
    )
    with pytest.raises(TypeError, match="^Sensitive features is of the wrong type."):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)
