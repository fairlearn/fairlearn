# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from . import group_accuracy_score, group_balanced_root_mean_squared_error
from . import group_fallout_rate, group_max_error
from . import group_mean_absolute_error, group_mean_overprediction
from . import group_mean_squared_error, group_mean_squared_log_error
from . import group_mean_underprediction, group_median_absolute_error
from . import group_mean_prediction
from . import group_miss_rate, group_precision_score, group_r2_score
from . import group_recall_score, group_roc_auc_score, group_root_mean_squared_error
from . import group_selection_rate, group_specificity_score, group_zero_one_loss

from . import GroupMetricResult

from fairlearn.metrics._input_manipulations import _convert_to_ndarray_1d

_GROUPS_NOT_SEQUENTIAL_INTEGERS = "The unique values of the groups property must be sequential integers from zero"  # noqa: E501
_GROUP_NAMES_MSG = "The group_names property must be a list of strings"
_METRICS_KEYS_MSG = "Keys for metrics dictionary must be strings"
_METRICS_VALUES_MSG = "Values for metrics dictionary must be of type GroupMetricResult"


class GroupMetricSet:
    """Class to hold a collection of GroupMetricResult objects."""

    BINARY_CLASSIFICATION = 'binary_classification'
    REGRESSION = 'regression'
    _allowed_model_types = frozenset([BINARY_CLASSIFICATION, REGRESSION])

    # The following keys need to match those of _metric_methods in
    # _fairlearn_dashboard.py
    # Issue 269 is about unifying the two sets
    GROUP_ACCURACY_SCORE = "accuracy_score"
    GROUP_BALANCED_ROOT_MEAN_SQUARED_ERROR = "balanced_root_mean_squared_error"
    GROUP_FALLOUT_RATE = "fallout_rate"
    GROUP_MAX_ERROR = "max_error"
    GROUP_MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    GROUP_MEAN_OVERPREDICTION = "overprediction"
    GROUP_MEAN_PREDICTION = "average"
    GROUP_MEAN_SQUARED_ERROR = "mean_squared_error"
    GROUP_MEAN_SQUARED_LOG_ERROR = "6d106114-4433-40a2-b091-8983ab540a53"
    GROUP_MEAN_UNDERPREDICTION = "underprediction"
    GROUP_MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    GROUP_MISS_RATE = "miss_rate"
    GROUP_PRECISION_SCORE = "precision_score"
    GROUP_R2_SCORE = "r2_score"
    GROUP_RECALL_SCORE = "recall_score"
    GROUP_ROC_AUC_SCORE = "balanced_accuracy_score"
    GROUP_ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
    GROUP_SELECTION_RATE = "selection_rate"
    GROUP_SPECIFICITY_SCORE = "specificity_score"
    GROUP_ZERO_ONE_LOSS = "zero_one_loss"

    BINARY_CLASSIFICATION_METRICS = {}
    BINARY_CLASSIFICATION_METRICS[GROUP_ACCURACY_SCORE] = group_accuracy_score
    BINARY_CLASSIFICATION_METRICS[GROUP_FALLOUT_RATE] = group_fallout_rate
    BINARY_CLASSIFICATION_METRICS[GROUP_MEAN_OVERPREDICTION] = group_mean_overprediction
    BINARY_CLASSIFICATION_METRICS[GROUP_MEAN_UNDERPREDICTION] = group_mean_underprediction
    BINARY_CLASSIFICATION_METRICS[GROUP_MISS_RATE] = group_miss_rate
    BINARY_CLASSIFICATION_METRICS[GROUP_PRECISION_SCORE] = group_precision_score
    BINARY_CLASSIFICATION_METRICS[GROUP_RECALL_SCORE] = group_recall_score
    BINARY_CLASSIFICATION_METRICS[GROUP_ROC_AUC_SCORE] = group_roc_auc_score
    BINARY_CLASSIFICATION_METRICS[GROUP_SELECTION_RATE] = group_selection_rate
    BINARY_CLASSIFICATION_METRICS[GROUP_SPECIFICITY_SCORE] = group_specificity_score

    REGRESSION_METRICS = {}
    REGRESSION_METRICS[GROUP_BALANCED_ROOT_MEAN_SQUARED_ERROR] = group_balanced_root_mean_squared_error  # noqa:E501
    REGRESSION_METRICS[GROUP_MAX_ERROR] = group_max_error
    REGRESSION_METRICS[GROUP_MEAN_ABSOLUTE_ERROR] = group_mean_absolute_error
    REGRESSION_METRICS[GROUP_MEAN_OVERPREDICTION] = group_mean_overprediction
    REGRESSION_METRICS[GROUP_MEAN_PREDICTION] = group_mean_prediction
    REGRESSION_METRICS[GROUP_MEAN_SQUARED_ERROR] = group_mean_squared_error
    REGRESSION_METRICS[GROUP_MEAN_SQUARED_LOG_ERROR] = group_mean_squared_log_error
    REGRESSION_METRICS[GROUP_MEAN_UNDERPREDICTION] = group_mean_underprediction
    REGRESSION_METRICS[GROUP_MEDIAN_ABSOLUTE_ERROR] = group_median_absolute_error
    REGRESSION_METRICS[GROUP_R2_SCORE] = group_r2_score
    REGRESSION_METRICS[GROUP_ROOT_MEAN_SQUARED_ERROR] = group_root_mean_squared_error
    REGRESSION_METRICS[GROUP_ZERO_ONE_LOSS] = group_zero_one_loss

    def __init__(self):
        self._model_type = None
        self._y_true = None
        self._y_pred = None
        self._groups = None
        self._group_names = None
        self._group_title = None
        self._metrics = None

    @property
    def model_type(self):
        """Return the type of model being described."""
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        if value not in GroupMetricSet._allowed_model_types:
            msg_format = "model_type '{0}' not in {1}"
            msg = msg_format.format(value, sorted(
                list(GroupMetricSet._allowed_model_types)))
            raise ValueError(msg)
        self._model_type = value

    @property
    def y_true(self):
        """Return the array of y_true values."""
        return self._y_true

    @y_true.setter
    def y_true(self, value):
        self._y_true = _convert_to_ndarray_1d(value)

    @property
    def y_pred(self):
        """Return the array of y_pred values."""
        return self._y_pred

    @y_pred.setter
    def y_pred(self, value):
        self._y_pred = _convert_to_ndarray_1d(value)

    @property
    def groups(self):
        """Return the array of group values.

        The unique values will always be sequential integers from zero.
        """
        return self._groups

    @groups.setter
    def groups(self, value):
        unique_values = np.unique(value)
        if not np.array_equal(unique_values, np.array(range(len(unique_values)))):
            raise ValueError(_GROUPS_NOT_SEQUENTIAL_INTEGERS)
        self._groups = _convert_to_ndarray_1d(value)

    @property
    def group_names(self):
        """Return the group_names list."""
        return self._group_names

    @group_names.setter
    def group_names(self, value):
        if not type(value) is list:
            raise ValueError(_GROUP_NAMES_MSG)
        value_types = set(type(v) for v in value)
        if value_types != {str}:
            raise ValueError(_GROUP_NAMES_MSG)

        self._group_names = value

    @property
    def group_title(self):
        """Return the title for the groups."""
        return self._group_title

    @group_title.setter
    def group_title(self, value):
        self._group_title = str(value)

    @property
    def metrics(self):
        """Return the GUID-GroupMetricResult dictionary of group metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        key_types = set(type(k) for k in value.keys())
        if key_types != {str}:
            raise ValueError(_METRICS_KEYS_MSG)
        value_types = set(type(v) for v in value.values())
        if value_types != {GroupMetricResult}:
            raise ValueError(_METRICS_VALUES_MSG)
        self._metrics = value

    def compute(self, y_true, y_pred, groups, model_type=BINARY_CLASSIFICATION):
        """Compute the default metrics.

        This will also automatically provide a remapping of the groups so that
        the unique values of the stored `groups` property will be sequential
        integers from zero.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_type = model_type

        unique_groups = sorted(list(np.unique(groups)))
        self.group_names = [str(x) for x in unique_groups]
        self.groups = [unique_groups.index(x) for x in groups]

        function_dict = None
        if self.model_type == GroupMetricSet.BINARY_CLASSIFICATION:
            function_dict = GroupMetricSet.BINARY_CLASSIFICATION_METRICS
        elif self.model_type == GroupMetricSet.REGRESSION:
            function_dict = GroupMetricSet.REGRESSION_METRICS
        else:
            raise RuntimeError("Cannot get here")

        computed_metrics = {}
        for metric_key, metric_function in function_dict.items():
            computed_metrics[metric_key] = metric_function(
                self.y_true, self.y_pred, self.groups)

        self.metrics = computed_metrics

    def __eq__(self, other):
        """Compare two `GroupMetricSet` objects for equality."""
        result = NotImplemented
        if isinstance(other, GroupMetricSet):
            result = self.model_type == other.model_type
            result = result and self.group_title == other.group_title
            result = result and np.array_equal(self.y_true, other.y_true)
            result = result and np.array_equal(self.y_pred, other.y_pred)
            result = result and np.array_equal(self.groups, other.groups)
            result = result and np.array_equal(self.group_names, other.group_names)
            result = result and self.metrics == other.metrics
        return result

    def __ne__(self, other):
        """Compare two `GroupMetricSet` objects for inequality."""
        are_equal = self.__eq__(other)
        if are_equal is NotImplemented:
            return are_equal
        else:
            return not are_equal
