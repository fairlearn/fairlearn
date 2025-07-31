# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import re
from contextlib import nullcontext as does_not_raise

import narwhals.stable.v1 as nw
import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from fairlearn.preprocessing import CorrelationRemover


@parametrize_with_checks(
    [
        CorrelationRemover(sensitive_feature_ids=[]),
        CorrelationRemover(sensitive_feature_ids=[0]),
    ],
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize(
    ["sensitive_feature_ids", "X", "expected_result"],
    [
        (
            [0],
            np.array(
                [
                    [0, 0, 1, 1],
                    [1, 1, 2, 2],
                    [0.1, 0.2, 1.2, 1.1],
                ]
            ).T,
            np.array([[1.5, 0.6], [1.5, 0.7], [1.5, 0.7], [1.5, 0.6]]),
        ),
    ],
)
def test_linear_dependence_array(sensitive_feature_ids, X, expected_result) -> None:
    X_tfm = CorrelationRemover(sensitive_feature_ids=sensitive_feature_ids).fit_transform(X)
    np.testing.assert_array_almost_equal(expected_result, X_tfm)


def test_linear_dependence_dataframe(constructor) -> None:
    sensitive_feature_ids = "a"
    data = {"a": [0, 0, 1, 1], "b": [1, 1, 2, 2], "c": [0.1, 0.2, 1.2, 1.1]}
    X_ = constructor(data)
    expected_result = np.array([[1.5, 0.6], [1.5, 0.7], [1.5, 0.7], [1.5, 0.6]])

    X_tfm = CorrelationRemover(sensitive_feature_ids=sensitive_feature_ids).fit_transform(X_)
    np.testing.assert_array_almost_equal(expected_result, X_tfm)


@pytest.mark.parametrize(
    ["alpha", "expected_result"],
    [
        (0.0, np.array([[2, 3], [5, 6], [8, 9]])),
        (0.5, np.array([[3.5, 4.5], [5, 6], [6.5, 7.5]])),
        (1.0, np.array([[5, 6], [5, 6], [5, 6]])),
    ],
)
def test_correlation_remover_handles_alpha(alpha, expected_result) -> None:
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    X_tfm = CorrelationRemover(sensitive_feature_ids=[0], alpha=alpha).fit_transform(X)
    np.testing.assert_array_almost_equal(expected_result, X_tfm)


@pytest.mark.parametrize("X", [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])])
def test_correlation_remover_handles_default_sensitive_features(X) -> None:
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    X_tfm = CorrelationRemover().fit_transform(X)
    np.testing.assert_array_equal(np.array(X), X_tfm)


def test_correlation_remover_handles_default_sensitive_features_dataframe(constructor) -> None:
    data = {"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]}
    X = constructor(data)
    X_tfm = CorrelationRemover().fit_transform(X)
    np.testing.assert_array_equal(nw.from_native(X, eager_only=True).to_numpy(), X_tfm)


@pytest.mark.parametrize(
    ["sensitive_feature_ids", "X", "expectation"],
    [
        (None, np.array([]), does_not_raise()),
        ([], np.array([[1, 2], [3, 4]]), does_not_raise()),
        ([0], np.array([[1, 2], [3, 4]]), does_not_raise()),
        ((1, 0), np.array([[1, 2]]), does_not_raise()),
        (
            [-1, 0, 3, 4],
            np.array([[1, 2], [3, 4]]),
            pytest.raises(
                ValueError, match=re.escape("Columns [-1, 3, 4] not found in the input data.")
            ),
        ),
    ],
)
def test__check_sensitive_features_in_X_array(sensitive_feature_ids, X, expectation) -> None:
    remover = CorrelationRemover(sensitive_feature_ids=sensitive_feature_ids)

    with expectation:
        remover._check_sensitive_features_in_X(X)


@pytest.mark.parametrize(
    ["sensitive_feature_ids", "data", "expectation"],
    [
        (["a", "b"], {"a": [1, 2], "b": [3, 4], "c": [5, 6]}, does_not_raise()),
        (
            ["a", "b", "c", "d"],
            {"a": [1, 2], "b": [3, 4]},
            pytest.raises(
                ValueError, match=re.escape("Columns ['c', 'd'] not found in the input data.")
            ),
        ),
    ],
)
def test__check_sensitive_features_in_X_array_dataframe(
    constructor, sensitive_feature_ids, data, expectation
) -> None:
    X = constructor(data)
    remover = CorrelationRemover(sensitive_feature_ids=sensitive_feature_ids)

    with expectation:
        remover._check_sensitive_features_in_X(X)
