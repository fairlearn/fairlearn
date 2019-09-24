# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class MetricResult:
    def __init__(self):
        # The 'metric' property is the given metric evaluated without regard to group
        self.metric = None
        # The 'group_metric' array contains one entry for each group passed to the
        # evaluation routine. The groups are required to be integers, starting from
        # zero. group_metric[i] contains the metric evaluated just for members of
        # the group marked with i
        self.group_metric = []
