# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Defines the Fairlearn dashboard class."""

from warnings import warn


class FairlearnDashboard(object):
    def __init__(
            self, *,
            sensitive_features,
            y_true, y_pred,
            sensitive_feature_names=None):
        warn("The FairlearnDashboard has moved from Fairlearn to the "
             "raiwidgets package. "
             "For more information on how to use it refer to "
             "https://github.com/microsoft/responsible-ai-widgets. "
             "Instead, Fairlearn now provides some of the existing "
             "functionality through matplotlib-based visualizations.")
