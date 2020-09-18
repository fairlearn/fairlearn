# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

import test_othermlpackages.package_test_common as ptc

from fairlearn.reductions import DemographicParity

try:
    import xgboost as xgb
except ImportError:
    pytestmark = pytest.mark.skip(reason="XGBoost not installed")


def test_expgrad_classification():
    estimator = xgb.XGBClassifier()
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


def test_gridsearch_classification():
    estimator = xgb.XGBClassifier()
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)
