# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


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
        """Return the minimum value of the metric in the ``by_group`` dictionary."""
        return min(self.by_group.values())

    @property
    def maximum(self):
        """Return the maximum value of the metric in the ``by_group`` dictionary."""
        return max(self.by_group.values())

    @property
    def argmin_set(self):
        """Return the set of groups corresponding to the ``minimum``.

        This will be a set of keys to tbe ``by_group`` dictionary.
        """
        return set([k for k, v in self.by_group.items() if v == self.minimum])

    @property
    def argmax_set(self):
        """Return the set of groups corresponding to the ``minimum``.

        This will be a set of keys to the ``by_group`` dictionary.
        """
        return set([k for k, v in self.by_group.items() if v == self.maximum])

    @property  # noqa: A003
    def range(self):  # noqa: A003
        """Return the value of :code:`maximum-minimum`."""
        return self.maximum - self.minimum

    @property
    def range_ratio(self):
        """Return the value of :code:`minimum/maximum`.

        This is only set if the metric is a scalar.
        """
        if self.minimum < 0:
            return np.nan
        elif self.maximum == 0:
            # We have min=max=0
            return 1
        else:
            return self.minimum / self.maximum

    @range_ratio.setter
    def range_ratio(self, value):
        self._range_ratio = value

    def __eq__(self, other):
        """Compare two `GroupMetricResult` objects for equality."""
        result = NotImplemented
        if isinstance(other, GroupMetricResult):
            if isinstance(self.overall, np.ndarray) and isinstance(other.overall, np.ndarray):
                result = np.array_equal(self.overall, other.overall)
                result = result and self.by_group.keys() == other.by_group.keys()
                for k in self.by_group.keys():
                    result = result and np.array_equal(self.by_group[k], other.by_group[k])
            elif isinstance(self.overall, np.ndarray) or isinstance(other.overall, np.ndarray):
                # Note that the previous 'and' test means that only one
                # side of this 'or' can be true
                result = False
            else:
                result = self.overall == other.overall
                result = result and self.by_group == other.by_group
        return result

    def __ne__(self, other):
        """Compare two `GroupMetricResult` objects for inequality."""
        are_equal = self.__eq__(other)
        if are_equal is NotImplemented:
            return are_equal
        else:
            return not are_equal
