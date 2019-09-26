# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GroupMetricResult:
    def __init__(self):
        # The 'overall' property is the given metric evaluated without regard to group
        self.overall = None
        # The 'by_group' dictionary contains the metric for each group found in the
        # input
        self.by_group = {}
        # The following two properties list the minimum and maximum metric values in
        # the by_group dictionary
        self.min_over_groups = None
        self.max_over_groups = None
        # The following two properties are the set of groups which have the minimum
        # and maximum values for the metric
        self.argmin_groups = None
        self.argmax_groups = None
        # The value of max_over_groups - min_over_groups
        self.range_over_groups = None
        # The value of min_over_groups / max_over_groups
        self.range_ratio_over_groups = None
