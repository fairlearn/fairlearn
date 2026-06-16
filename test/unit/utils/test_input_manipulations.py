# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from fairlearn.utils._input_manipulations import (
    _convert_to_ndarray_1d,
    _convert_to_ndarray_and_squeeze,
)


@pytest.mark.parametrize(
    "X, expected",
    [
        ([0, 1, 2], np.array([0, 1, 2])),
        ([[0], [1]], np.array([0, 1])),
        ([[0, 1]], np.array([0, 1])),
        ([[[1]]], np.array([1])),
    ],
)
def test_convert_to_ndarray_and_squeeze(X, expected: NDArray):
    result = _convert_to_ndarray_and_squeeze(X)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "X, expected",
    [
        ([0, 1, 2], np.array([0, 1, 2])),
        ([[4, 5]], np.array([4, 5])),
        ([[5], [7]], np.array([5, 7])),
    ],
)
def test_convert_to_ndarray_1d(X, expected: NDArray):
    result = _convert_to_ndarray_1d(X)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "X, expectation",
    [
        (
            [[1, 2], [3, 4]],
            pytest.raises(
                ValueError, match="Supplied input array has more than one non-trivial dimension"
            ),
        )
    ],
)
def test_convert_to_ndarray_1d_raises_exception(X, expectation: AbstractContextManager):
    with expectation:
        _convert_to_ndarray_1d(X)


def test_ThresholdOptimize_handles_X_with_ndims_greater_than_2() -> None:
    class DummyEstimator(BaseEstimator, ClassifierMixin):
        def fit(self, X, y, **kwargs):
            self.fitted_ = True
            return self

        def predict(self, X):
            return np.array([0, 1] * (len(X) // 2))

        def predict_proba(self, X):
            return np.array([[0.6, 0.4], [0.4, 0.6]] * (len(X) // 2))

    X = np.random.rand(10, 5, 3)
    y = np.array([0, 1] * 5)
    sensitive_features = ["A"] * 5 + ["B"] * 5

    exponentiated_gradient = ExponentiatedGradient(
        estimator=DummyEstimator(), constraints=DemographicParity(), max_iter=10
    )

    with does_not_raise():
        exponentiated_gradient.fit(X, y, sensitive_features=sensitive_features)
        exponentiated_gradient.predict(X)
