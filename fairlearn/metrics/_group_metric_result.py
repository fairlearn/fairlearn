# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GroupMetricResult:
    """Class to hold the result of a grouped metric, produced by calling
    the :func:`metric_by_group` function.
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
        self._argmin_set = None
        self._argmax_set = None
        # The value of maximum - minimum
        self._range = None
        # The value of minimum / maximum
        self._range_ratio = None

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

    @property
    def argmin_set(self):
        """If ``minimum`` is set, this is the set of
        groups (that is, keys in the ``by_group``
        dictionary) corresponding to the minimum value
        of the metric.
        """
        return self._argmin_set

    @argmin_set.setter
    def argmin_set(self, value):
        self._argmin_set = value

    @property
    def argmax_set(self):
        """If ``maximum`` is set, this is the set of
        groups (that is, keys in the ``by_group``
        dictionary) corresponding to the maximum value
        of the metric.
        """
        return self._argmax_set

    @argmax_set.setter
    def argmax_set(self, value):
        self._argmax_set = value

    @property
    def range(self):
        """If ``maximum`` and ``minimum`` are set, this
        will be set to the difference between them
        """
        return self._range

    @range.setter
    def range(self, value):
        self._range = value

    @property
    def range_ratio(self):
        """If ``maximum`` and ``minimum`` are set, this
        will be set to the ratio ``minimum/maximum``
        """
        return self._range_ratio

    @range_ratio.setter
    def range_ratio(self, value):
        self._range_ratio = value
