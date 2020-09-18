# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

import test_othermlpackages.package_test_common as ptc

from fairlearn.reductions import DemographicParity

try:
    import lightgbm as lgb
except ImportError:
    pytestmark = pytest.mark.skip(reason="LightGBM not installed")


def test_expgrad_classification():
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 10,
        'max_depth': 3
    }
    estimator = lgb.LGBMClassifier(**lgb_params)
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


def test_gridsearch_classification():
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 10,
        'max_depth': 3
    }
    estimator = lgb.LGBMClassifier(**lgb_params)
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)
