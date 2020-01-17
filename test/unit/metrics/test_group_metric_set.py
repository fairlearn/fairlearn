# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import GroupMetricSet


def test_model_types():
    assert GroupMetricSet.BINARY_CLASSIFICATION == "binary_classification"
