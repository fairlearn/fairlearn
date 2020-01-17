# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.metrics import GroupMetricSet, GroupMetricResult


def test_model_type_property():
    target = GroupMetricSet()
    target.mode_type = GroupMetricSet.BINARY_CLASSIFICATION
    assert target.mode_type == "binary_classification"


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
    expected = "Keys not integers"
    assert exception_context.value.args[0] == expected


def test_group_names_values_not_string():
    target = GroupMetricSet()
    with pytest.raises(ValueError) as exception_context:
        target.group_names = {0: 1, 2: 3}
    expected = "Values not strings"
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
    expected = "Keys not strings"
    assert exception_context.value.args[0] == expected


def test_metrics_values_not_groupmetricresult():
    target = GroupMetricSet()
    with pytest.raises(ValueError) as exception_context:
        target.metrics = {"a": 0}
    expected = "Values not GroupMetricResults"
    assert exception_context.value.args[0] == expected
