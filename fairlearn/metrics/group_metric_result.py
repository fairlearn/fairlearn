# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GroupMetricResult:
    """Class to hold the result of a grouped metric, produced by calling
    fairlearn.metrics.metric_by_group.

    :members:

    *overall* The result of applying the metric to all the data

    *by_group* A dictionary of the metric applied to each subgroup. The
    keys the unique values of the `group_membership` array supplied
    to the metric.

    If the result of the metric is a scalar, then the following
    will also be defined:

    *minimum* The minimum value of the metric from the `by_group`
    dictionary

    *maximum* The maximum value of the metric from the `by_group`
    dictionary

    *argmin_set* The set of groups which correspond to the `minimum`

    *argmax_set* The set of groups which correspond to the `maximum`

    *range* The value of `maximum` - `minimum`

    *range_ratio* The value of `minimum` / `maximum`
    """

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
