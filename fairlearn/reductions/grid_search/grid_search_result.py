# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GridSearchResult:
    def __init__(self, model, lagrange_multiplier, quality_metric_value):
        self.model = model
        self.lagrange_multiplier = lagrange_multiplier
        self.quality_metric_value = quality_metric_value
