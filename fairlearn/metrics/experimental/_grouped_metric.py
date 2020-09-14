# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd


class GroupedMetric:
    def __init__(self, metric_functions, y_true, y_pred, sensitive_features, sample_param_names, params):
        pass

    @property
    def overall(self):
        return self._overall

    @property
    def by_group(self):
        return self._by_group