# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics as metrics
from test.unit.input_convertors import conversions_for_1d


# ===========================================================


def mock_func(y_true, y_pred):
    return np.sum(y_true)


def mock_func_weight(y_true, y_pred, sample_weight):
    return np.sum(np.multiply(y_true, sample_weight))


def mock_func_matrix_return(y_true, y_pred):
    return np.ones([len(y_true), sum(y_pred)])


class TestMetricByGroup:
    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
    def test_smoke(self, transform_y_a, transform_y_p, transform_gid):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 1, 1])

        result = metrics.metric_by_group(mock_func, y_a, y_p, gid)

        assert result.overall == 5
        assert len(result.by_group) == 2
        assert result.by_group[0] == 2
        assert result.by_group[1] == 3
        assert result.minimum == 2
        assert result.argmin_set == {0}
        assert result.maximum == 3
        assert result.argmax_set == {1}
        assert result.range == 1
        assert result.range_ratio == pytest.approx(0.6666666667)

    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
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
        assert result.minimum == 1
        assert result.argmin_set == {a, b}
        assert result.maximum == 3
        assert result.argmax_set == {c}
        assert result.range == 2
        assert result.range_ratio == pytest.approx(0.33333333333333)

    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
    def test_matrix_metric(self, transform_y_a, transform_y_p, transform_gid):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, a, a, b, b, c, c, c])

        result = metrics.metric_by_group(mock_func_matrix_return, y_a, y_p, gid)

        assert np.array_equal(result.overall, np.ones([8, 5]))
        assert np.array_equal(result.by_group[a], np.ones([3, 2]))
        assert np.array_equal(result.by_group[b], np.ones([2, 2]))
        assert np.array_equal(result.by_group[c], np.ones([3, 1]))

    def test_matrix_metric_other_properties(self):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [a, a, a, b, b, c, c, c]

        result = metrics.metric_by_group(mock_func_matrix_return, y_a, y_p, gid)

        # Other fields should fail
        with pytest.raises(ValueError):
            _ = result.minimum
        with pytest.raises(ValueError):
            _ = result.maximum
        with pytest.raises(ValueError):
            _ = result.argmin_set
        with pytest.raises(ValueError):
            _ = result.argmax_set
        with pytest.raises(ValueError):
            _ = result.range
        with pytest.raises(ValueError):
            _ = result.range_ratio

    @pytest.mark.parametrize("transform_s_w", conversions_for_1d)
    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
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
        assert result.minimum == 2
        assert result.argmin_set == {0, 1}
        assert result.maximum == 6
        assert result.argmax_set == {2}
        assert result.range == 4
        assert result.range_ratio == pytest.approx(0.33333333333333)

    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
    def test_true_predict_length_mismatch(self, transform_y_a, transform_y_p):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0])
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_group(mock_func_weight, y_a, y_p, gid, s_w)

        expected = "Array y_pred is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
    def test_true_group_length_mismatch(self, transform_y_a, transform_gid):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = transform_gid([0, 0, 0, 0, 1, 1, 2])
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_group(mock_func_weight, y_a, y_p, gid, s_w)

        expected = "Array group_membership is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    @pytest.mark.parametrize("transform_s_w", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
    def test_true_weight_length_mismatch(self, transform_y_a, transform_s_w):
        y_a = transform_y_a([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = [0, 0, 0, 0, 1, 1, 2, 3]
        s_w = transform_s_w([1, 1, 1, 1, 2, 2, 3])

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.metric_by_group(mock_func_weight, y_a, y_p, gid, s_w)

        expected = "Array sample_weight is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    def test_negative_results(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 0, 1, 1, 1]

        def negative_results(y_true, y_pred):
            return -(len(y_true) + len(y_pred))

        result = metrics.metric_by_group(negative_results, y_a, y_p, gid)

        assert result.overall == -16
        assert result.by_group[0] == -10
        assert result.by_group[1] == -6
        assert result.minimum == -10
        assert result.maximum == -6
        assert result.range == 4
        assert np.isnan(result.range_ratio)

    def test_metric_results_zero(self):
        y_a = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 0, 1, 1, 1]

        def zero_results(y_true, y_pred):
            # Arrays will always be same length
            return len(y_true)-len(y_pred)

        result = metrics.metric_by_group(zero_results, y_a, y_p, gid)

        assert result.overall == 0
        assert result.by_group[0] == 0
        assert result.by_group[1] == 0
        assert result.minimum == 0
        assert result.maximum == 0
        assert result.range == 0
        # Following is special case
        assert result.range_ratio == 1

    def test_single_element_input(self):
        y_t = [0]
        y_p = [0]
        gid = [0]
        s_w = [0]

        def sum_lengths(y_true, y_pred, sample_weight):
            return len(y_true) + len(y_pred) + len(sample_weight)

        result = metrics.metric_by_group(sum_lengths, y_t, y_p, gid, sample_weight=s_w)
        assert result.overall == 3
        assert result.by_group[0] == 3
        assert result.minimum == 3
        assert result.maximum == 3
        assert result.range == 0
        assert result.range_ratio == 1

    def test_groups_only_one_element(self):
        y_t = [1, 2]
        y_p = [1, 2]
        gid = [0, 1]

        def sum_lengths(y_true, y_pred):
            return len(y_true) + len(y_pred)

        result = metrics.metric_by_group(sum_lengths, y_t, y_p, gid)
        assert result.overall == 4
        assert result.by_group[0] == 2
        assert result.by_group[1] == 2
        assert result.minimum == 2
        assert result.maximum == 2
        assert result.range == 0
        assert result.range_ratio == 1


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
        assert result.minimum == 2
        assert result.maximum == 3
        assert result.argmin_set == {0}
        assert result.argmax_set == {1}
        assert result.range == 1
        assert result.range_ratio == pytest.approx(0.66666666667)

    @pytest.mark.parametrize("transform_s_w", conversions_for_1d)
    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_a", conversions_for_1d)
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
        assert result.minimum == 1
        assert result.maximum == 21
        assert result.argmin_set == {a, z}
        assert result.argmax_set == {c}
        assert result.range == 20
        assert result.range_ratio == pytest.approx(1.0/21.0)
