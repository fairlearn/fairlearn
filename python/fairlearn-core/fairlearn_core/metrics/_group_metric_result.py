# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GroupMetricResult:
    """Class to hold the result of a grouped metric.

    Grouped metrics are produced by the :func:`metric_by_group`
    function.
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
        """Return the metric calculated over the entire dataset."""
        return self._overall

    @overall.setter
    def overall(self, value):
        self._overall = value

    @property
    def by_group(self):
        """Return the metric calculated for each sub-group in the dataset.

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
        """Return the minimum value of the metric in the ``by_group`` dictionary.

        This is only set if the metric is a scalar.
        """
        return self._minimum

    @minimum.setter
    def minimum(self, value):
        self._minimum = value

    @property
    def maximum(self):
        """Return the maximum value of the metric in the ``by_group`` dictionary.

        This is only set if the metric is a scalar.
        """
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        self._maximum = value

    @property
    def argmin_set(self):
        """Return the set of groups corresponding to the ``minimum``.

        This is only set if the metric is a scalar, and will be
        a set of keys to tbe ``by_group`` dictionary.
        """
        return self._argmin_set

    @argmin_set.setter
    def argmin_set(self, value):
        self._argmin_set = value

    @property
    def argmax_set(self):
        """Return the set of groups corresponding to the ``minimum``.

        This is only set if the metric is a scalar, and will be
        a set of keys to tbe ``by_group`` dictionary.
        """
        return self._argmax_set

    @argmax_set.setter
    def argmax_set(self, value):
        self._argmax_set = value

    @property  # noqa: A003
    def range(self):
        """Return the value of :code:`maximum-minimum`.

        This is only set if the metric is a scalar.
        """
        return self._range

    @range.setter  # noqa: A003
    def range(self, value):
        self._range = value

    @property
    def range_ratio(self):
        """Return the value of :code:`minimum/maximum`.

        This is only set if the metric is a scalar.
        """
        return self._range_ratio

    @range_ratio.setter
    def range_ratio(self, value):
        self._range_ratio = value
