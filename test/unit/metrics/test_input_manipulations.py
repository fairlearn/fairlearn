# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics._input_manipulations as fmim


class TestEnsure1DNDArray:
    def test_simple_list(self):
        X = [0, 1, 2]

        result = fmim._ensure_1d_ndarray(X, "X")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 2

    def test_multi_rows(self):
        X = [[0], [1]]

        result = fmim._ensure_1d_ndarray(X, "X")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 0
        assert result[1] == 1

    def test_multi_columns(self):
        X = [[0, 1]]

        result = fmim._ensure_1d_ndarray(X, "X")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 0
        assert result[1] == 1

    def test_single_element(self):
        X = [[[1]]]

        result = fmim._ensure_1d_ndarray(X, "X")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 1

    def test_multid_rejected_simple(self):
        X = [[0, 1], [2, 3]]

        with pytest.raises(ValueError) as exception_context:
            _ = fmim._ensure_1d_ndarray(X, "X")

        expected = "'X' has more than one dimension longer than 1"
        assert exception_context.value.args[0] == expected

    def test_multid_rejected_complex(self):
        X = [[[[0], [1]], [[2], [3]]]]
        assert np.asarray(X).shape == (1, 2, 2, 1)

        with pytest.raises(ValueError) as exception_context:
            _ = fmim._ensure_1d_ndarray(X, "X")

        expected = "'X' has more than one dimension longer than 1"
        assert exception_context.value.args[0] == expected
