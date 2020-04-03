# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from sklearn import preprocessing

from . import group_accuracy_score, group_balanced_root_mean_squared_error
from . import group_fallout_rate, group_max_error
from . import group_mean_absolute_error, group_mean_overprediction
from . import group_mean_squared_error, group_mean_squared_log_error
from . import group_mean_underprediction, group_median_absolute_error
from . import group_mean_prediction
from . import group_miss_rate, group_precision_score, group_r2_score
from . import group_recall_score, group_roc_auc_score, group_root_mean_squared_error
from . import group_selection_rate, group_specificity_score, group_zero_one_loss
from ._input_manipulations import _convert_to_ndarray_and_squeeze

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
_PREDICTION_REGRESSION = 'regression'
_MODEL_NAMES = 'modelNames'
_SCHEMA = 'schemaType'
_DASHBOARD_DICTIONARY = 'dashboardDictionary'
_VERSION = 'schemaVersion'

BINARY_CLASSIFICATION = 'binary_classification'
REGRESSION = 'regression'
_allowed_prediction_types = frozenset([BINARY_CLASSIFICATION, REGRESSION])

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


def _process_sensitive_features(sensitive_features):
    """Convert the dictionary into the required list."""
    unsorted_features = []
    for column_name, column in sensitive_features.items():
        nxt = dict()
        nxt[_FEATURE_BIN_NAME] = column_name

        np_column = _convert_to_ndarray_and_squeeze(column)
        le = preprocessing.LabelEncoder()

        # Since these will likely be JSON serialised we
        # need to make sure we have Python ints and not
        # numpy ints
        nxt[_BIN_VECTOR] = [int(x) for x in list(le.fit_transform(np_column))]
        nxt[_BIN_LABELS] = [str(x) for x in le.classes_]

        unsorted_features.append(nxt)
    result = sorted(unsorted_features, key=lambda x: x[_FEATURE_BIN_NAME])
    return result


def _process_predictions(predictions):
    """Convert the dictionary into two lists."""
    names = []
    preds = []
    for model_name in sorted(predictions):
        names.append(model_name)
        y_p = _convert_to_ndarray_and_squeeze(predictions[model_name])
        preds.append(y_p.tolist())
    return names, preds


def _create_group_metric_set(y_true,
                             predictions,
                             sensitive_features,
                             prediction_type):
    """Create a dictionary matching the Dashboard's cache."""
    result = dict()
    result[_SCHEMA] = _DASHBOARD_DICTIONARY
    result[_VERSION] = 0

    if prediction_type not in _allowed_prediction_types:
        msg_format = "prediction_type '{0}' not in {1}"
        msg = msg_format.format(prediction_type, sorted(
            list(_allowed_prediction_types)))
        raise ValueError(msg)

    function_dict = None
    if prediction_type == BINARY_CLASSIFICATION:
        result[_PREDICTION_TYPE] = _PREDICTION_BINARY_CLASSIFICATION
        function_dict = BINARY_CLASSIFICATION_METRICS
    elif prediction_type == REGRESSION:
        result[_PREDICTION_TYPE] == _PREDICTION_REGRESSION
        function_dict = REGRESSION_METRICS
    else:
        raise NotImplementedError(
            "No support yet for {0}".format(prediction_type))

    # Sort out y_true
    _yt = _convert_to_ndarray_and_squeeze(y_true)
    result[_Y_TRUE] = _yt.tolist()

    # Sort out predictions
    result[_MODEL_NAMES], result[_Y_PRED] = _process_predictions(predictions)

    # Sort out the sensitive features
    result[_PRECOMPUTED_BINS] = _process_sensitive_features(sensitive_features)

    result[_PRECOMPUTED_METRICS] = []
    for g in result[_PRECOMPUTED_BINS]:
        by_prediction_list = []
        for prediction in result[_Y_PRED]:
            metric_dict = dict()
            for metric_key, metric_func in function_dict.items():
                gmr = metric_func(result[_Y_TRUE], prediction, g[_BIN_VECTOR])
                curr_dict = dict()
                curr_dict[_GLOBAL] = gmr.overall
                curr_dict[_BINS] = list(gmr.by_group.values())
                metric_dict[metric_key] = curr_dict
            by_prediction_list.append(metric_dict)
        result[_PRECOMPUTED_METRICS].append(by_prediction_list)

    return result
