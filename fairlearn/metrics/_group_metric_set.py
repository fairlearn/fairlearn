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

_GROUP_NAMES_MSG = "The group_names property must be a list of strings"
_METRICS_KEYS_MSG = "Keys for metrics dictionary must be strings"
_METRICS_VALUES_MSG = "Values for metrics dictionary must be of type GroupMetricResult"

_ARRAYS_NOT_SAME_LENGTH = "Lengths of y_true, y_pred and groups must match"
_BIN_MISMATCH_FOR_METRIC = "The groups for metric {0} do not match the groups property"
_GROUP_NAMES_BAD_COUNT = "Count of group_names not the same as the number of unique groups"

_Y_TRUE = 'trueY'
_Y_PRED = 'predictedY'
_PRECOMPUTED_METRICS = 'precomputedMetrics'
_GLOBAL = 'global'
_BINS = 'bins'
_PRECOMPUTED_BINS = 'precomputedFeatureBins'
_BIN_VECTOR = 'binVector'
_BIN_LABELS = 'binLabels'
_FEATURE_BIN_NAME = 'featureBinName'
_PREDICTION_TYPE = 'predictionType'
_PREDICTION_BINARY_CLASSIFICATION = 'binaryClassification'
_MODEL_NAMES = 'modelNames'
_SCHEMA = 'schemaType'
_GROUP_METRIC_SET = 'groupMetricSet'
_VERSION = 'schemaVersion'

_UNSUPPORTED_MODEL_TYPE = "The specified model_type of '{0}' is not supported"
_DICT_TOO_MANY_Y_PRED = 'Too many y_pred values in dictionary'

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


def create_group_metric_set(model_type,
                            y_true,
                            y_preds,
                            group_memberships,
                            model_titles=None,
                            group_titles=None,
                            extra_metrics=None):
    """Create a dictionary matching the Dashboard's cache."""
    if extra_metrics is not None:
        raise NotImplementedError("No support for extra_metrics yet")

    # We could consider checking that the length of y_preds matches model_titles
    # and that the length of group_memberships matches group_titles

    result = dict()
    result[_SCHEMA] = _GROUP_METRIC_SET
    result[_VERSION] = 0

    if model_type not in _allowed_model_types:
        msg_format = "model_type '{0}' not in {1}"
        msg = msg_format.format(model_type, sorted(
            list(_allowed_model_types)))
        raise ValueError(msg)

    function_dict = None
    if model_type == BINARY_CLASSIFICATION:
        result[_PREDICTION_TYPE] = _PREDICTION_BINARY_CLASSIFICATION
        function_dict = BINARY_CLASSIFICATION_METRICS
    else:
        raise NotImplementedError("No support yet for regression")

    _yt = np.asarray(y_true)
    result[_Y_TRUE] = _yt.tolist()

    result[_Y_PRED] = []
    result[_PRECOMPUTED_METRICS] = []
    result[_PRECOMPUTED_BINS] = []
    result[_MODEL_NAMES] = []
    for g, group_membership in enumerate(group_memberships):
        _gm = np.asarray(group_membership).tolist()
        _unique_groups = sorted(list(np.unique(_gm)))
        group_names = [str(x) for x in _unique_groups]
        groups = [_unique_groups.index(x) for x in _gm]
        bin_dict = {_BIN_VECTOR: groups, _BIN_LABELS: group_names}
        if group_titles is not None:
            bin_dict[_FEATURE_BIN_NAME] = group_titles[g]
        result[_PRECOMPUTED_BINS].append(bin_dict)

        model_list = []
        for m, model_pred in enumerate(y_preds):
            _yp = np.asarray(model_pred).tolist()

            # Only record each y_pred and model name once
            if g == 0:
                result[_Y_PRED].append(_yp)
                if model_titles is not None:
                    result[_MODEL_NAMES].append(model_titles[m])

            metric_dict = dict()
            for metric_key, metric_func in function_dict.items():
                gmr = metric_func(_yt, _yp, groups)
                curr_dict = dict()
                curr_dict[_GLOBAL] = gmr.overall
                curr_dict[_BINS] = list(gmr.by_group.values())
                metric_dict[metric_key] = curr_dict
            model_list.append(metric_dict)
        result[_PRECOMPUTED_METRICS].append(model_list)

    return result
