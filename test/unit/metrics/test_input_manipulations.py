# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics._input_manipulations as fmim


class TestConvertToNDArrayAndSqueeze:
    def test_simple_list(self):
        X = [0, 1, 2]

        result = fmim._convert_to_ndarray_and_squeeze(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 2

    def test_multi_rows(self):
        X = [[0], [1]]

        result = fmim._convert_to_ndarray_and_squeeze(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 0
        assert result[1] == 1

    def test_multi_columns(self):
        X = [[0, 1]]

        result = fmim._convert_to_ndarray_and_squeeze(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 0
        assert result[1] == 1

    def test_single_element(self):
        X = [[[1]]]

        result = fmim._convert_to_ndarray_and_squeeze(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 1


class TestConvertToNDArray1D:
    def test_simple_list(self):
        X = [0, 1, 2]

        result = fmim._convert_to_ndarray_1d(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 2

    def test_simple_nested_list(self):
        X = [[4, 5]]

        result = fmim._convert_to_ndarray_1d(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 4
        assert result[1] == 5

    def test_simple_transposed_list(self):
        X = [[5], [7]]

        result = fmim._convert_to_ndarray_1d(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result[0] == 5
        assert result[1] == 7

    def test_2d_raises_exception(self):
        X = [[1, 2], [3, 4]]

        with pytest.raises(ValueError) as exception_context:
            _ = fmim._convert_to_ndarray_1d(X)

        expected = "Supplied input array has more than one non-trivial dimension"
        assert exception_context.value.args[0] == expected
