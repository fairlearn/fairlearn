# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

import fairlearn.metrics as metrics

# ===========================================================

# Conversions from Python lists to our supported datatypes


def identity(X):
    return X


def tondarray(X):
    return np.asarray(X)


def tondarray2d(X):
    # ndarray where second dimension is of length 1
    arr = np.asarray(X)
    arr = np.expand_dims(arr, 1)
    assert len(arr.shape) == 2
    return arr


def topandasseries(X):
    return pd.Series(X)


def topandasdf(X):
    return pd.DataFrame(X)


supported_conversions = [identity, tondarray, tondarray2d, topandasseries, topandasdf]

# ===========================================================


def mock_func(y_true, y_pred):
    return np.sum(y_true)


def mock_func_weight(y_true, y_pred, sample_weight):
    return np.sum(np.multiply(y_true, sample_weight))


class TestMetricByGroup:
    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_smoke(self, transform_y_a, transform_y_p, transform_gid):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 1, 1])

        result = metrics.metric_by_group(mock_func, y_a, y_p, gid)

        assert result.overall == 5
        assert len(result.by_group) == 2
        assert result.by_group[0] == 2
        assert result.by_group[1] == 3
        assert result.min_over_groups == 2
        assert result.argmin_groups == {0}
        assert result.max_over_groups == 3
        assert result.argmax_groups == {1}
        assert result.range_over_groups == 1
        assert result.range_ratio_over_groups == 1.5

    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_string_groups(self, transform_y_a, transform_y_p, transform_gid):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, a, a, b, b, c, c, c])

        result = metrics.metric_by_group(mock_func, y_a, y_p, gid)

        assert result.overall == 5
        assert len(result.by_group) == 3
        assert result.by_group[a] == 1
        assert result.by_group[b] == 1
        assert result.by_group[c] == 3
        assert result.min_over_groups == 1
        assert result.argmin_groups == {a, b}
        assert result.max_over_groups == 3
        assert result.argmax_groups == {c}
        assert result.range_over_groups == 2
        assert result.range_ratio_over_groups == 3

    @pytest.mark.parametrize("transform_s_w", supported_conversions)
    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_with_weights(self, transform_y_a, transform_y_p, transform_gid, transform_s_w):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 2, 2])
        s_w = transform_s_w([1, 1, 1, 1, 2, 2, 3, 3])

        result = metrics.metric_by_group(mock_func_weight, y_a, y_p, gid, sample_weight=s_w)

        assert result.overall == 10
        assert len(result.by_group) == 3
        assert result.by_group[0] == 2
        assert result.by_group[1] == 2
        assert result.by_group[2] == 6
        assert result.min_over_groups == 2
        assert result.argmin_groups == {0, 1}
        assert result.max_over_groups == 6
        assert result.argmax_groups == {2}
        assert result.range_over_groups == 4
        assert result.range_ratio_over_groups == 3

    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_true_predict_length_mismatch(self, transform_y_a, transform_y_p):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0])
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_group(mock_func_weight, y_a, y_p, gid, s_w)

        expected = "Array y_pred is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_true_group_length_mismatch(self, transform_y_a, transform_gid):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = transform_gid([0, 0, 0, 0, 1, 1, 2])
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_group(mock_func_weight, y_a, y_p, gid, s_w)

        expected = "Array group_data is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    @pytest.mark.parametrize("transform_s_w", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_true_weight_length_mismatch(self, transform_y_a, transform_s_w):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = [0, 0, 0, 0, 1, 1, 2, 3]
        s_w = transform_s_w([1, 1, 1, 1, 2, 2, 3])

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_group(mock_func_weight, y_a, y_p, gid, s_w)

        expected = "Array sample_weight is not the same size as y_true"
        assert exception_context.value.args[0] == expected


class TestMakeGroupMetric:
    def test_smoke(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 1, 1]

        grouped_metric_func = metrics.make_group_metric(mock_func)
        result = grouped_metric_func(y_a, y_p, gid)
        assert result.overall == 5
        assert len(result.by_group) == 2
        assert result.by_group[0] == 2
        assert result.by_group[1] == 3
        assert result.min_over_groups == 2
        assert result.max_over_groups == 3
        assert result.argmin_groups == {0}
        assert result.argmax_groups == {1}
        assert result.range_over_groups == 1
        assert result.range_ratio_over_groups == 1.5

    @pytest.mark.parametrize("transform_s_w", supported_conversions)
    @pytest.mark.parametrize("transform_gid", supported_conversions)
    @pytest.mark.parametrize("transform_y_p", supported_conversions)
    @pytest.mark.parametrize("transform_y_a", supported_conversions)
    def test_keys_and_weights(self, transform_y_a, transform_y_p, transform_gid, transform_s_w):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        z = "something_longer"
        y_a = transform_y_a([0, 1, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, z, a, b, b, c, c, c])
        s_w = transform_s_w([1, 1, 1, 5, 5, 7, 7, 7])

        grouped_metric_func = metrics.make_group_metric(mock_func_weight)
        result = grouped_metric_func(y_a, y_p, gid, s_w)
        assert result.overall == 28
        assert len(result.by_group) == 4
        assert result.by_group[a] == 1
        assert result.by_group[b] == 5
        assert result.by_group[c] == 21
        assert result.by_group[z] == 1
        assert result.min_over_groups == 1
        assert result.max_over_groups == 21
        assert result.argmin_groups == {a, z}
        assert result.argmax_groups == {c}
        assert result.range_over_groups == 20
        assert result.range_ratio_over_groups == 21
