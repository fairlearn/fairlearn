# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest
from fairlearn.postprocessing._threshold_operation import ThresholdOperation


def test_predict_from_operation_less():
    classifier = ThresholdOperation('<', 0.5)
    assert classifier(-10000) == 1
    assert classifier(0) == 1
    assert classifier(0.5) == 0
    assert classifier(1) == 0
    assert classifier(10000) == 0


def test_predict_from_operation_more():
    classifier = ThresholdOperation('>', 0.5)
    assert classifier(-10000) == 0
    assert classifier(0) == 0
    assert classifier(0.5) == 0
    assert classifier(1) == 1
    assert classifier(10000) == 1


def test_predict_from_operation_invalid_operator():
    with pytest.raises(ValueError, match="Unrecognized operator: ="):
        ThresholdOperation('=', 0.5)
