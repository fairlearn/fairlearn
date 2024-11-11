# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from contextlib import AbstractContextManager

import numpy as np
import pytest
from numpy.typing import NDArray

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
