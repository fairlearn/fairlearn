# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GroupMetricResult:
    """Class to hold the result of a grouped metric, produced by calling
    fairlearn.metrics.metric_by_group.
    """

    def __init__(self):
        # The 'overall' property is the given metric evaluated without regard to group
        self._overall = None
        # The 'by_group' dictionary contains the metric for each group found in the
        # input
        self._by_group = {}
        # The following two properties list the minimum and maximum metric values in
        # the by_group dictionary
        self._minimum = None
        self._maximum = None
        # The following two properties are the set of groups which have the minimum
        # and maximum values for the metric
        self.argmin_set = None
        self.argmax_set = None
        # The value of maximum - minimum
        self.range = None
        # The value of minimum / maximum
        self.range_ratio = None

    @property
    def overall(self):
        """Gets the value of the metric calculated
        over the entire dataset
        """
        return self._overall

    @overall.setter
    def overall(self, value):
        self._overall = value

    @property
    def by_group(self):
        """Gets the value of the metric calculated for each sub-group
        in the dataset.
        This is a dictionary whose keys are the unique members of
        the ``group_membership`` data. The corresponding values are
        the result of applying the metric function to the set of
        ``y_true`` and ``y_pred`` entries for each key.
        """
        return self._by_group

    @by_group.setter
    def by_group(self, value):
        self._by_group = value

    @property
    def minimum(self):
        """Gets the minimum value of the metric found in the
        ``by_group`` dictionary, if the value is a scalar.
        Otherwise, this will not be set.
        """
        return self._minimum

    @minimum.setter
    def minimum(self, value):
        self._minimum = value

    @property
    def maximum(self):
        """Gets the maxumum value of the metric found in the
        ``by_group`` dictionary, if the values is a scalar.
        Otherwise, this will not be set
        """
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        self._maximum = value
