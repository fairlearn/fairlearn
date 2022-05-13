# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

from sklearn.linear_model import LogisticRegression

from fairlearn.linear_model._constrained_logistic import (
    ConstrainedLogisticRegression,
)


def test_unconstrained_vs_normal_lr(data_X, data_y, constraints=None):
    normal_lr = LogisticRegression(random_state=0).fit(data_X, data_y)
    unconstrained = ConstrainedLogisticRegression(
        random_state=0, constraints=constraints
    ).fit(data_X, data_y)

    assert normal_lr.score(data_X, data_y) == unconstrained.score(
        data_X, data_y
    )
    # TODO: Fixme
    assert np.all(
        normal_lr.predict_proba(data_X) == unconstrained.predict_proba(data_X)
    )


def test_one_sensitive_feature(data_X, data_y, data_single_sf):
    n_samples = len(data_y)
    classes = np.unique(data_y)
    n_classes = classes.shape[0]

    X = np.hstack((np.array(data_X), np.array(data_single_sf))).astype(
        "object"
    )

    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0
    )
    predicted = clf.fit(X, data_y, sensitive_feature_ids=[1]).predict(data_X)
    assert predicted.shape == (n_samples,)

    probabilities = clf.predict_proba(data_X)
    assert probabilities.shape == (n_samples, n_classes)


def test_two_sensitive_features(data_X, data_y, data_multiple_sf):
    n_samples = len(data_y)
    classes = np.unique(data_y)
    n_classes = classes.shape[0]

    X = np.hstack((np.array(data_X), np.array(data_multiple_sf))).astype(
        "object"
    )

    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0
    )
    predicted = clf.fit(X, data_y, sensitive_feature_ids=[1, 2]).predict(
        data_X
    )
    assert predicted.shape == (n_samples,)

    probabilities = clf.predict_proba(data_X)
    assert probabilities.shape == (n_samples, n_classes)
