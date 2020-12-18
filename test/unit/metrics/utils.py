# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import fairlearn.metrics as metrics


def _get_raw_MetricFrame():
    # Gets an uninitialised MetricFrame for testing purposes
    return metrics.MetricFrame.__new__(metrics.MetricFrame)
