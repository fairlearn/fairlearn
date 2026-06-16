# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.base import BaseEstimator, ClassifierMixin

from fairlearn.utils._common import _filter_kwargs, _get_soft_predictions


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """This is only used to test _get_soft_predictions."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([1, 2, 1])

    def predict_proba(self, X):
        return np.array(
            [
                [0.2, 0.6],
                [0.8, 0.4],
                [0.3, 0.7],
            ]
        )

    def decision_function(self, X):
        return np.array([0.5, -0.7, 0.9])


@pytest.mark.parametrize(
    "method, y_true",
    [
        ("predict", [1, 2, 1]),
        ("predict_proba", [0.6, 0.4, 0.7]),
        ("decision_function", [0.5, -0.7, 0.9]),
        ("auto", [0.6, 0.4, 0.7]),
    ],
)
def test_soft_predictions(method, y_true):
    clf = DummyClassifier()

    output = _get_soft_predictions(clf, None, method)
    assert_array_almost_equal(y_true, output)


@pytest.mark.parametrize(
    "func, kwargs, expected",
    [
        (lambda a, b=0: (a, b), {}, {}),
        (lambda a, **kwargs: a, {}, {}),
        (lambda a, b=0: (a, b), {"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1}),
        (lambda a, **kwargs: a, {"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 1}),
    ],
)
def test_filter_kwargs(func, kwargs, expected):
    """Test that `_filter_kwargs` correctly filters the kwargs accepted by a function or
    method."""
    filtered_kwargs = _filter_kwargs(func=func, kwargs=kwargs)
    assert filtered_kwargs == expected
