# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.metrics import plot_disparities_in_performance
from .conftest import quickstart_setup


# These are just smoke tests.  For ways to improve testing,
# see https://github.com/fairlearn/fairlearn/pull/289#issuecomment-587170978
def test_plot_disparities_in_performance():
    (y_true, y_pred, sensitive_features) = quickstart_setup()
    plot_disparities_in_performance(y_true, y_pred, sensitive_features, show_plot=False)
