# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import fairlearn.metrics.experimental as metrics


def _get_raw_GroupedMetric():
    # Gets an uninitialised GroupedMetric for testing purposes
    return metrics.GroupedMetric.__new__(metrics.GroupedMetric)
