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
        self.minimum = None
        self.maximum = None
        # The following two properties are the set of groups which have the minimum
        # and maximum values for the metric
        self.argmin_set = None
        self.argmax_set = None
        # The value of maximum - minimum
        self.range = None
        # The value of minimum / maximum
        self.range_ratio = None
