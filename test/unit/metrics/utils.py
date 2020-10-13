# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import fairlearn.metrics as metrics


def _get_raw_MetricsFrame():
    # Gets an uninitialised MetricsFrame for testing purposes
    return metrics.MetricsFrame.__new__(metrics.MetricsFrame)
