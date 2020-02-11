# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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

_GROUP_NAMES_KEYS_MSG = "Keys for group_names dictionary must be integers"
_GROUP_NAMES_VALUES_MSG = "Values for group_names dictionary must be strings"
_METRICS_KEYS_MSG = "Keys for metrics dictionary must be strings"
_METRICS_VALUES_MSG = "Values for metrics dictionary must be of type GroupMetricResult"


class GroupMetricSet:
    """Class to hold a collection of GroupMetricResult objects."""

    BINARY_CLASSIFICATION = 'binary_classification'
    REGRESSION = 'regression'
    _allowed_model_types = frozenset([BINARY_CLASSIFICATION, REGRESSION])

    GROUP_ACCURACY_SCORE = "4c038715-e944-4902-9eb2-973a8fd6b51e"
    GROUP_BALANCED_ROOT_MEAN_SQUARED_ERROR = "1e37c604-8a14-4d57-8fe6-2c88c06ae3c0"
    GROUP_FALLOUT_RATE = "dacb8b66-bd5c-4b71-b8eb-410792aad99d"
    GROUP_MAX_ERROR = "83abd9f8-bef1-4602-a87c-5fa55b429261"
    GROUP_MEAN_ABSOLUTE_ERROR = "12c23096-536e-40c5-b29d-261a9053a2a9"
    GROUP_MEAN_OVERPREDICTION = "c20da0b0-1412-451e-977d-951ff815b4f7"
    GROUP_MEAN_PREDICTION = "7c6af4a9-c148-4890-8acb-465d2bbfe0bb"
    GROUP_MEAN_SQUARED_ERROR = "f36190c7-7b05-4837-97cf-f7fb3da19752"
    GROUP_MEAN_SQUARED_LOG_ERROR = "6d106114-4433-40a2-b091-8983ab540a53"
    GROUP_MEAN_UNDERPREDICTION = "e7e2ca86-b7b2-4eb3-8cc3-b63fa0a8e37f"
    GROUP_MEDIAN_ABSOLUTE_ERROR = "66e6c952-f83f-48b8-a884-931f511dd7da"
    GROUP_MISS_RATE = "5f4e4f9c-f09b-4aee-998a-0e9eeba208c6"
    GROUP_PRECISION_SCORE = "d2dcd0a6-3aba-4d91-8715-2655fb37674b"
    GROUP_R2_SCORE = "445c1bec-9c6c-4085-bdea-26e37cea6194"
    GROUP_RECALL_SCORE = "1e72002d-449d-40d9-b94c-5f053fd26032"
    GROUP_ROC_AUC_SCORE = "5322378d-1839-4d91-8bb3-e54d295ed3cd"
    GROUP_ROOT_MEAN_SQUARED_ERROR = "e8abde62-5511-4a67-b2c0-08b344227d7e"
    GROUP_SELECTION_RATE = "adad0b7b-55b6-4e2f-9583-fdecc2ad9cb9"
    GROUP_SPECIFICITY_SCORE = "53649e54-3d2b-495c-827e-bf4b201d7597"
    GROUP_ZERO_ONE_LOSS = "3c2805a7-04b5-4b19-baa9-33928fe3d121"

    BINARY_CLASSIFICATION_METRICS = {}
    BINARY_CLASSIFICATION_METRICS[GROUP_ACCURACY_SCORE] = group_accuracy_score
    BINARY_CLASSIFICATION_METRICS[GROUP_FALLOUT_RATE] = group_fallout_rate
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
        """Return the array of group values."""
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = _convert_to_ndarray_1d(value)

    @property
    def group_names(self):
        """Return the group_names dictionary."""
        return self._group_names

    @group_names.setter
    def group_names(self, value):
        key_types = set(type(k) for k in value.keys())
        if key_types != {int}:
            raise ValueError(_GROUP_NAMES_KEYS_MSG)
        value_types = set(type(v) for v in value.values())
        if value_types != {str}:
            raise ValueError(_GROUP_NAMES_VALUES_MSG)

        self._group_names = value

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

    def compute(self, y_true, y_pred, groups, model_type=BINARY_CLASSIFICATION, group_names=None):
        """Compute the default metrics."""
        self.y_true = y_true
        self.y_pred = y_pred
        self.groups = groups
        self.model_type = model_type
        if group_names is not None:
            self.group_names = group_names

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
