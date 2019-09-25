# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class MetricResult:
    def __init__(self):
        # The 'metric' property is the given metric evaluated without regard to group
        self.metric = None
        # The 'group_metric' dictionary contains the metric for each group found in the
        # input
        self.group_metric = {}
        self.min_metric = None
        self.max_metric = None
        self.metric_range = None
        self.metric_range_ratio = None
