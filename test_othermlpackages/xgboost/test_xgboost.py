# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

try:
    import xgboost as xgb
except ImportError:
    pytestmark = pytest.mark.skip(reason="XGBoost not installed")


def test_expgrad_classification():
    print(xgb.__name__)  # Shut up warning
    assert False


def test_gridsearch_classification():
    assert False
