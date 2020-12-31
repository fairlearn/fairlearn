# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from .conftest import quickstart_setup
from fairlearn.metrics import plot_disparities_in_metric
from fairlearn.metrics._group_metric_set import BINARY_CLASSIFICATION_METRICS, \
    REGRESSION_METRICS
import pytest


# These are just smoke tests.  For ways to improve testing,
# see https://github.com/fairlearn/fairlearn/pull/289#issuecomment-587170978
@pytest.mark.parametrize(
    "metric", list(BINARY_CLASSIFICATION_METRICS.values()) +
    list(REGRESSION_METRICS.values()))
def test_plot_disparities_in_metric(metric):
    (y_true, y_pred, sensitive_features) = quickstart_setup()
    plot_disparities_in_metric(metric, y_true, y_pred, sensitive_features, show_plot=False)
