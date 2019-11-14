# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

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
