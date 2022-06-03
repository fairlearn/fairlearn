# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

from fairlearn.linear_model._constrained_logistic import (
    ConstrainedLogisticRegression,
    _process_sensitive_features,
    _ohe_sensitive_features,
    _get_constraint_list_cov,
    _process_covariance_bound_dict,
)
from .conftest import (
    cov_bound_dict_right,
    cov_bound_dict_wrong,
    mulitple_sf,
    mulitple_sf_dataframe,
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
    num_sens_features = data_multiple_sf.shape[1]
    covariance_bound = [0] * (num_sens_features + 1)
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=covariance_bound, n_jobs=-1
    )
    with pytest.raises(
        ValueError, match="^Number of covariance bound values can not exceed"
    ):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_too_little_cov_bound_values(data_X, data_y, data_multiple_sf):
    sensitive_features = np.hstack((data_multiple_sf, data_multiple_sf))
    num_sens_features = sensitive_features.shape[1]
    covariance_bound = [0] * (num_sens_features - 1)
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=covariance_bound, n_jobs=-1
    )
    with pytest.raises(
        ValueError,
        match="^Number of covariance bound values is higher than 1 but lower than",
    ):
        clf.fit(data_X, data_y, sensitive_features=sensitive_features)


@pytest.mark.parametrize("cov_bound", [cov_bound_dict_right, cov_bound_dict_wrong])
@pytest.mark.parametrize("sensitive_features", [mulitple_sf, mulitple_sf_dataframe])
def test_cov_bound_dict(cov_bound, sensitive_features):
    if isinstance(sensitive_features, np.ndarray):
        with pytest.raises(
            TypeError,
            match="^The sensitive features are of type",
        ):
            _process_covariance_bound_dict(cov_bound, sensitive_features)
    elif isinstance(sensitive_features, pd.DataFrame):
        if not sorted(sensitive_features.columns) == sorted(cov_bound.keys()):
            with pytest.raises(
                ValueError,
                match="^The keys in the covariance bound dictionary do not match",
            ):
                _process_covariance_bound_dict(cov_bound, sensitive_features)
    else:
        cov_bound = _process_covariance_bound_dict(cov_bound, sensitive_features)
        assert isinstance(cov_bound, list)


def test_wrong_solver(data_X, data_y, data_multiple_sf):
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", solver="lbfgs", covariance_bound=0, n_jobs=-1
    )
    with pytest.raises(
        ValueError, match="^Constrained Logistic Regression supports only solvers in"
    ):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_wrong_penalty(data_X, data_y, data_multiple_sf):
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", penalty="l1", covariance_bound=0, n_jobs=-1
    )
    with pytest.raises(
        ValueError, match="^Constrained Logistic Regression supports only penalties in"
    ):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_wrong_multi_class(data_X, data_y, data_multiple_sf):
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity",
        multi_class="auto",
        covariance_bound=0,
        n_jobs=-1,
    )
    with pytest.raises(ValueError, match="^multi_class should be 'ovr'"):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_mismatch_X_sf_rows(data_X, data_y, data_multiple_sf):
    data_multiple_sf = data_multiple_sf[:-1, :]
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0, n_jobs=-1
    )
    with pytest.raises(ValueError, match="^X has [0-9]+ instances while"):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)


def test_process_sensitive_features(data_multiple_sf):
    sensitive_features, num_sens_features = _process_sensitive_features(
        data_multiple_sf
    )
    assert isinstance(sensitive_features, (np.ndarray, pd.DataFrame))
    assert isinstance(num_sens_features, int)


def test_ohe_sensitive_features(data_single_sf):
    sensitive_features, num_sens_features = _process_sensitive_features(data_single_sf)
    sensitive_features, categories = _ohe_sensitive_features(sensitive_features)

    # First assert is to check if the data is one-hot-encoded
    assert (
        sensitive_features.sum(axis=1) - np.ones(sensitive_features.shape[0])
    ).sum() == 0
    assert isinstance(categories, list)


def test_get_constraint_list_cov(
    data_X,
    data_multiple_sf,
):
    sensitive_features, num_sens_features = _process_sensitive_features(
        data_multiple_sf
    )
    sensitive_features, categories = _ohe_sensitive_features(sensitive_features)
    covariance_bound = [0] * num_sens_features
    constraints = _get_constraint_list_cov(
        data_X,
        sensitive_features,
        categories,
        covariance_bound,
    )
    for constraint in constraints:
        assert "type" in constraint
        assert "fun" in constraint
        assert "args" in constraint


def test_sf_wrong_type(data_X, data_y, data_multiple_sf):
    data_multiple_sf = tuple(data_multiple_sf)
    clf = ConstrainedLogisticRegression(
        constraints="demographic_parity", covariance_bound=0, n_jobs=-1
    )
    with pytest.raises(TypeError, match="^Sensitive features is of the wrong type."):
        clf.fit(data_X, data_y, sensitive_features=data_multiple_sf)
