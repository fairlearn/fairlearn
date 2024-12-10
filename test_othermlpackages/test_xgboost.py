# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest
import sklearn
from packaging.version import parse

from fairlearn.reductions import DemographicParity

from . import package_test_common as ptc

xgb = pytest.importorskip("xgboost")


def _should_skip_test():
    return parse(sklearn.__version__) >= parse("1.6.0")


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_expgrad_classification():
    estimator = xgb.XGBClassifier()
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_gridsearch_classification():
    estimator = xgb.XGBClassifier()
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_thresholdoptimizer_classification():
    estimator = xgb.XGBClassifier()

    ptc.run_thresholdoptimizer_classification(estimator)
