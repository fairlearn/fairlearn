# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import GroupMetricSet


def test_model_type():
    target = GroupMetricSet()
    target.mode_type = GroupMetricSet.BINARY_CLASSIFICATION
    assert target.mode_type == "binary_classification"
