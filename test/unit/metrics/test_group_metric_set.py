# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.metrics import group_accuracy_score, group_balanced_root_mean_squared_error
from fairlearn.metrics import GroupMetricSet, GroupMetricResult


def test_model_type_property():
    target = GroupMetricSet()

    for mt in GroupMetricSet._allowed_model_types:
        target.model_type = mt
        assert target.model_type == mt


def test_model_type_not_allowed():
    target = GroupMetricSet()

    with pytest.raises(ValueError) as exception_context:
        target.model_type = "Something Random"
    expected = "model_type 'Something Random' not in ['binary_classification', 'regression']"
    assert exception_context.value.args[0] == expected


def test_y_true():
    target = GroupMetricSet()
    target.y_true = [1, 2, 3]
    assert isinstance(target.y_true, np.ndarray)
    assert np.array_equal(target.y_true, [1, 2, 3])


def test_y_pred():
    target = GroupMetricSet()
    target.y_pred = [1, 2]
    assert isinstance(target.y_pred, np.ndarray)
    assert np.array_equal(target.y_pred, [1, 2])


def test_groups():
    target = GroupMetricSet()
    target.groups = [4, 5, 6]
    assert isinstance(target.groups, np.ndarray)
    assert np.array_equal(target.groups, [4, 5, 6])


def test_group_names():
    target = GroupMetricSet()
    target.group_names = {0: 'a', 1: 'b'}
    assert target.group_names[0] == 'a'
    assert target.group_names[1] == 'b'


def test_group_names_keys_not_int():
    target = GroupMetricSet()
    with pytest.raises(ValueError) as exception_context:
        target.group_names = {'a': 'b', 'c': 'd'}
    expected = "Keys for group_names dictionary must be integers"
    assert exception_context.value.args[0] == expected


def test_group_names_values_not_string():
    target = GroupMetricSet()
    with pytest.raises(ValueError) as exception_context:
        target.group_names = {0: 1, 2: 3}
    expected = "Values for group_names dictionary must be strings"
    assert exception_context.value.args[0] == expected


def test_metrics():
    target = GroupMetricSet()
    my_metric = GroupMetricResult()
    my_metric.overall = 10222
    target.metrics = {"60bdb14d-83fb-4374-ab2e-4c371f22b21e": my_metric}
    assert target.metrics["60bdb14d-83fb-4374-ab2e-4c371f22b21e"].overall == 10222


def test_metrics_keys_not_string():
    target = GroupMetricSet()
    my_metric = GroupMetricResult()
    my_metric.overall = 10222
    with pytest.raises(ValueError) as exception_context:
        target.metrics = {0: my_metric}
    expected = "Keys for metrics dictionary must be strings"
    assert exception_context.value.args[0] == expected


def test_metrics_values_not_groupmetricresult():
    target = GroupMetricSet()
    with pytest.raises(ValueError) as exception_context:
        target.metrics = {"a": 0}
    expected = "Values for metrics dictionary must be of type GroupMetricResult"
    assert exception_context.value.args[0] == expected


Y_true = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
groups = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0]


def test_compute_binary():
    target = GroupMetricSet()

    target.compute(Y_true, Y_pred, groups, model_type=GroupMetricSet.BINARY_CLASSIFICATION)

    sample_expected = group_accuracy_score(Y_true, Y_pred, groups)

    assert np.array_equal(Y_true, target.y_true)
    assert np.array_equal(Y_pred, target.y_pred)
    assert np.array_equal(groups, target.groups)
    assert len(target.metrics) == 10
    assert target.metrics[GroupMetricSet.GROUP_ACCURACY_SCORE].overall == sample_expected.overall
    for g in np.unique(groups):
        assert (target.metrics[GroupMetricSet.GROUP_ACCURACY_SCORE].by_group[g] ==
                sample_expected.by_group[g])


def test_compute_regression():
    target = GroupMetricSet()

    target.compute(Y_true, Y_pred, groups, model_type=GroupMetricSet.REGRESSION)

    sample_expected = group_balanced_root_mean_squared_error(Y_true, Y_pred, groups)

    assert np.array_equal(Y_true, target.y_true)
    assert np.array_equal(Y_pred, target.y_pred)
    assert np.array_equal(groups, target.groups)
    assert len(target.metrics) == 12
    assert target.metrics[GroupMetricSet.GROUP_BALANCED_ROOT_MEAN_SQUARED_ERROR].overall == sample_expected.overall  # noqa: E501
    for g in np.unique(groups):
        assert (target.metrics[GroupMetricSet.GROUP_BALANCED_ROOT_MEAN_SQUARED_ERROR].by_group[g]
                == sample_expected.by_group[g])
