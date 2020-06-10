# Copyright (c) Microsoft Corporation and contributors.
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


def mock_func_extra_arg(y_true, y_pred, my_arg=1):
    return np.sum([my_arg * y for y in y_true])


def mock_func_matrix_return(y_true, y_pred):
    return np.ones([len(y_true), sum(y_pred)])


class TestGroupSummary:
    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_smoke(self, transform_y_t, transform_y_p, transform_gid):
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 1, 1])

        result = metrics.group_summary(mock_func, y_t, y_p, sensitive_features=gid)

        assert result.overall == 5
        assert len(result.by_group) == 2
        assert result.by_group[0] == 2
        assert result.by_group[1] == 3
        assert metrics.group_min_from_summary(result) == 2
        assert metrics.group_max_from_summary(result) == 3
        assert metrics.difference_from_summary(result) == 1
        assert metrics.ratio_from_summary(result) == pytest.approx(0.6666666667)

    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_smoke_extra_arg(self, transform_y_t, transform_y_p, transform_gid):
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 1, 1])

        # Run with the argument defaulted
        result = metrics.group_summary(mock_func_extra_arg,
                                       y_t,
                                       y_p,
                                       sensitive_features=gid)
        assert result.overall == 5
        assert len(result.by_group) == 2
        assert result.by_group[0] == 2
        assert result.by_group[1] == 3

        # Run with the argument speficied
        result = metrics.group_summary(mock_func_extra_arg,
                                       y_t,
                                       y_p,
                                       sensitive_features=gid,
                                       my_arg=2)
        assert result.overall == 10
        assert len(result.by_group) == 2
        assert result.by_group[0] == 4
        assert result.by_group[1] == 6

    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_string_groups(self, transform_y_t, transform_y_p, transform_gid):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, a, a, b, b, c, c, c])

        result = metrics.group_summary(mock_func, y_t, y_p, sensitive_features=gid)

        assert result.overall == 5
        assert len(result.by_group) == 3
        assert result.by_group[a] == 1
        assert result.by_group[b] == 1
        assert result.by_group[c] == 3
        assert metrics.group_min_from_summary(result) == 1
        assert metrics.group_max_from_summary(result) == 3
        assert metrics.difference_from_summary(result) == 2
        assert metrics.ratio_from_summary(result) == pytest.approx(0.33333333333333)

    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_matrix_metric(self, transform_y_t, transform_y_p, transform_gid):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, a, a, b, b, c, c, c])

        result = metrics.group_summary(mock_func_matrix_return, y_t, y_p, sensitive_features=gid)

        assert np.array_equal(result.overall, np.ones([8, 5]))
        assert np.array_equal(result.by_group[a], np.ones([3, 2]))
        assert np.array_equal(result.by_group[b], np.ones([2, 2]))
        assert np.array_equal(result.by_group[c], np.ones([3, 1]))

    def test_matrix_metric_other_properties(self):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        y_t = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [a, a, a, b, b, c, c, c]

        result = metrics.group_summary(mock_func_matrix_return, y_t, y_p, sensitive_features=gid)

        # Other fields should fail
        with pytest.raises(ValueError):
            _ = metrics.group_min_from_summary(result)
        with pytest.raises(ValueError):
            _ = metrics.group_max_from_summary(result)
        with pytest.raises(ValueError):
            _ = metrics.difference_from_summary(result)
        with pytest.raises(ValueError):
            _ = metrics.ratio_from_summary(result)

    @pytest.mark.parametrize("transform_s_w", conversions_for_1d)
    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_with_weights(self, transform_y_t, transform_y_p, transform_gid, transform_s_w):
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([0, 0, 0, 0, 1, 1, 2, 2])
        s_w = transform_s_w([1, 1, 1, 1, 2, 2, 3, 3])

        result = metrics.group_summary(
            mock_func_weight, y_t, y_p, sensitive_features=gid, sample_weight=s_w)

        assert result.overall == 10
        assert len(result.by_group) == 3
        assert result.by_group[0] == 2
        assert result.by_group[1] == 2
        assert result.by_group[2] == 6
        assert metrics.group_min_from_summary(result) == 2
        assert metrics.group_max_from_summary(result) == 6
        assert metrics.difference_from_summary(result) == 4
        assert metrics.ratio_from_summary(result) == pytest.approx(0.33333333333333)

    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_true_predict_length_mismatch(self, transform_y_t, transform_y_p):
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0])
        gid = [0, 0, 0, 0, 1, 1, 2, 2]
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.group_summary(
                mock_func_weight, y_t, y_p, sensitive_features=gid, sample_weight=s_w)

        expected = "Array y_pred is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_true_group_length_mismatch(self, transform_y_t, transform_gid):
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = transform_gid([0, 0, 0, 0, 1, 1, 2])
        s_w = [1, 1, 1, 1, 2, 2, 3, 3]

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.group_summary(
                mock_func_weight, y_t, y_p, sensitive_features=gid, sample_weight=s_w)

        expected = "Array sensitive_features is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    @pytest.mark.parametrize("transform_s_w", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_true_weight_length_mismatch(self, transform_y_t, transform_s_w):
        y_t = transform_y_t([0, 0, 1, 1, 0, 1, 1, 1])
        y_p = [0, 1, 1, 1, 1, 0, 0, 0]
        gid = [0, 0, 0, 0, 1, 1, 2, 3]
        s_w = transform_s_w([1, 1, 1, 1, 2, 2, 3])

        with pytest.raises(ValueError) as exception_context:
            _ = metrics.group_summary(
                mock_func_weight, y_t, y_p, sensitive_features=gid, sample_weight=s_w)

        expected = "Array sample_weight is not the same size as y_true"
        assert exception_context.value.args[0] == expected

    def test_negative_results(self):
        y_t = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 0, 1, 1, 1]

        def negative_results(y_true, y_pred):
            return -(len(y_true) + len(y_pred))

        result = metrics.group_summary(negative_results, y_t, y_p, sensitive_features=gid)

        assert result.overall == -16
        assert result.by_group[0] == -10
        assert result.by_group[1] == -6
        assert metrics.group_min_from_summary(result) == -10
        assert metrics.group_max_from_summary(result) == -6
        assert metrics.difference_from_summary(result) == 4
        assert np.isnan(metrics.ratio_from_summary(result))

    def test_metric_results_zero(self):
        y_t = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 0, 1, 1, 1]

        def zero_results(y_true, y_pred):
            # Arrays will always be same length
            return len(y_true)-len(y_pred)

        result = metrics.group_summary(zero_results, y_t, y_p, sensitive_features=gid)

        assert result.overall == 0
        assert result.by_group[0] == 0
        assert result.by_group[1] == 0
        assert metrics.group_min_from_summary(result) == 0
        assert metrics.group_max_from_summary(result) == 0
        assert metrics.difference_from_summary(result) == 0
        # Following is special case
        assert metrics.ratio_from_summary(result) == 1

    def test_single_element_input(self):
        y_t = [0]
        y_p = [0]
        gid = [0]
        s_w = [0]

        def sum_lengths(y_true, y_pred, sample_weight):
            return len(y_true) + len(y_pred) + len(sample_weight)

        result = metrics.group_summary(
            sum_lengths, y_t, y_p, sensitive_features=gid, sample_weight=s_w)
        assert result.overall == 3
        assert result.by_group[0] == 3
        assert metrics.group_min_from_summary(result) == 3
        assert metrics.group_max_from_summary(result) == 3
        assert metrics.difference_from_summary(result) == 0
        assert metrics.ratio_from_summary(result) == 1

    def test_groups_only_one_element(self):
        y_t = [1, 2]
        y_p = [1, 2]
        gid = [0, 1]

        def sum_lengths(y_true, y_pred):
            return len(y_true) + len(y_pred)

        result = metrics.group_summary(sum_lengths, y_t, y_p, sensitive_features=gid)
        assert result.overall == 4
        assert result.by_group[0] == 2
        assert result.by_group[1] == 2
        assert metrics.group_min_from_summary(result) == 2
        assert metrics.group_max_from_summary(result) == 2
        assert metrics.difference_from_summary(result) == 0
        assert metrics.ratio_from_summary(result) == 1


class TestMakeMetricGroupSummary:
    def test_smoke(self):
        y_t = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 1, 1]

        grouped_metric_func = metrics.make_metric_group_summary(mock_func)
        result = grouped_metric_func(y_t, y_p, sensitive_features=gid)
        assert result.overall == 5
        assert len(result.by_group) == 2
        assert result.by_group[0] == 2
        assert result.by_group[1] == 3
        assert metrics.group_min_from_summary(result) == 2
        assert metrics.group_max_from_summary(result) == 3
        assert metrics.difference_from_summary(result) == 1
        assert metrics.ratio_from_summary(result) == pytest.approx(0.66666666667)

    @pytest.mark.parametrize("transform_s_w", conversions_for_1d)
    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_keys_and_weights(self, transform_y_t, transform_y_p, transform_gid, transform_s_w):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        z = "something_longer"
        y_t = transform_y_t([0, 1, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, z, a, b, b, c, c, c])
        s_w = transform_s_w([1, 1, 1, 5, 5, 7, 7, 7])

        grouped_metric_func = metrics.make_metric_group_summary(mock_func_weight)
        result = grouped_metric_func(y_t, y_p, sensitive_features=gid, sample_weight=s_w)
        assert result.overall == 28
        assert len(result.by_group) == 4
        assert result.by_group[a] == 1
        assert result.by_group[b] == 5
        assert result.by_group[c] == 21
        assert result.by_group[z] == 1
        assert metrics.group_min_from_summary(result) == 1
        assert metrics.group_max_from_summary(result) == 21
        assert metrics.difference_from_summary(result) == 20
        assert metrics.ratio_from_summary(result) == pytest.approx(1.0/21.0)


class TestMakeDerivedMetric:
    def test_smoke(self):
        y_t = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 1, 1]

        metric_group_summary = metrics.make_metric_group_summary(mock_func)
        metric_group_min = metrics.make_derived_metric(
            metrics.group_min_from_summary, metric_group_summary)
        metric_group_max = metrics.make_derived_metric(
            metrics.group_max_from_summary, metric_group_summary)
        metric_difference = metrics.make_derived_metric(
            metrics.difference_from_summary, metric_group_summary)
        metric_ratio = metrics.make_derived_metric(
            metrics.ratio_from_summary, metric_group_summary)

        assert metric_group_min(y_t, y_p, sensitive_features=gid) == 2
        assert metric_group_max(y_t, y_p, sensitive_features=gid) == 3
        assert metric_difference(y_t, y_p, sensitive_features=gid) == 1
        assert metric_ratio(y_t, y_p, sensitive_features=gid) == pytest.approx(0.66666666667)

    def test_smoke_extra_arg(self):
        y_t = [0, 0, 1, 1, 0, 1, 1, 1]
        y_p = [0, 1, 1, 1, 1, 0, 0, 1]
        gid = [0, 0, 0, 0, 1, 1, 1, 1]

        metric_group_summary = metrics.make_metric_group_summary(mock_func_extra_arg)
        metric_group_min = metrics.make_derived_metric(
            metrics.group_min_from_summary, metric_group_summary)
        metric_group_max = metrics.make_derived_metric(
            metrics.group_max_from_summary, metric_group_summary)
        metric_difference = metrics.make_derived_metric(
            metrics.difference_from_summary, metric_group_summary)
        metric_ratio = metrics.make_derived_metric(
            metrics.ratio_from_summary, metric_group_summary)

        # Run with the extra argument defaulted
        assert metric_group_min(y_t, y_p, sensitive_features=gid) == 2
        assert metric_group_max(y_t, y_p, sensitive_features=gid) == 3
        assert metric_difference(y_t, y_p, sensitive_features=gid) == 1
        assert metric_ratio(y_t, y_p, sensitive_features=gid) == pytest.approx(0.66666666667)

        # Run with the extra argument set to something
        assert metric_group_min(y_t, y_p, sensitive_features=gid, my_arg=2) == 4
        assert metric_group_max(y_t, y_p, sensitive_features=gid, my_arg=2) == 6
        assert metric_difference(y_t, y_p, sensitive_features=gid, my_arg=2) == 2
        assert metric_ratio(y_t, y_p, sensitive_features=gid,
                            my_arg=2) == pytest.approx(0.66666666667)

    @pytest.mark.parametrize("transform_s_w", conversions_for_1d)
    @pytest.mark.parametrize("transform_gid", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_p", conversions_for_1d)
    @pytest.mark.parametrize("transform_y_t", conversions_for_1d)
    def test_keys_and_weights(self, transform_y_t, transform_y_p, transform_gid, transform_s_w):
        a = "ABC"
        b = "DEF"
        c = "GHI"
        z = "something_longer"
        y_t = transform_y_t([0, 1, 1, 1, 0, 1, 1, 1])
        y_p = transform_y_p([0, 1, 1, 1, 1, 0, 0, 1])
        gid = transform_gid([a, z, a, b, b, c, c, c])
        s_w = transform_s_w([1, 1, 1, 5, 5, 7, 7, 7])

        metric_group_summary = metrics.make_metric_group_summary(mock_func_weight)
        metric_group_min = metrics.make_derived_metric(
            metrics.group_min_from_summary, metric_group_summary)
        metric_group_max = metrics.make_derived_metric(
            metrics.group_max_from_summary, metric_group_summary)
        metric_difference = metrics.make_derived_metric(
            metrics.difference_from_summary, metric_group_summary)
        metric_ratio = metrics.make_derived_metric(
            metrics.ratio_from_summary, metric_group_summary)

        assert metric_group_min(y_t, y_p, sensitive_features=gid, sample_weight=s_w) == 1
        assert metric_group_max(y_t, y_p, sensitive_features=gid, sample_weight=s_w) == 21
        assert metric_difference(y_t, y_p, sensitive_features=gid, sample_weight=s_w) == 20
        assert metric_ratio(y_t, y_p,
                            sensitive_features=gid, sample_weight=s_w) == pytest.approx(1.0/21.0)
