# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GroupMetricResult:
    def __init__(self):
        # The 'metric' property is the given metric evaluated without regard to group
        self.overall = None
        # The 'group_metric' dictionary contains the metric for each group found in the
        # input
        self.by_group = {}
        self.min_over_groups = None
        self.max_over_groups = None
        self.argmin_groups = None
        self.argmax_groups = None
        self.range_over_groups = None
        self.range_ratio_over_groups = None
