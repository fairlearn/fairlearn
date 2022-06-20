# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import warnings

import numpy as np
import sklearn.metrics as skm
from sklearn import preprocessing

from ._extra_metrics import (
    _balanced_root_mean_squared_error,
    _mean_overprediction,
    _mean_underprediction,
    _root_mean_squared_error,
    count,
    false_negative_rate,
    false_positive_rate,
    mean_prediction,
    selection_rate,
    true_negative_rate,
)
from ._input_manipulations import _convert_to_ndarray_and_squeeze
from ._metric_frame import MetricFrame

_Y_TRUE = "trueY"
_Y_PRED = "predictedY"
_PRECOMPUTED_METRICS = "precomputedMetrics"
_GLOBAL = "global"
_BINS = "bins"
_PRECOMPUTED_BINS = "precomputedFeatureBins"
_BIN_VECTOR = "binVector"
_BIN_LABELS = "binLabels"
_FEATURE_BIN_NAME = "featureBinName"
_PREDICTION_TYPE = "predictionType"
_PREDICTION_BINARY_CLASSIFICATION = "binaryClassification"
_PREDICTION_PROBABILITY = "probability"
_PREDICTION_REGRESSION = "regression"
_MODEL_NAMES = "modelNames"
_SCHEMA = "schemaType"
_DASHBOARD_DICTIONARY = "dashboardDictionary"
_VERSION = "schemaVersion"

BINARY_CLASSIFICATION = "binary_classification"
PROBABILITY = "probability"
REGRESSION = "regression"
_allowed_prediction_types = frozenset([BINARY_CLASSIFICATION, PROBABILITY, REGRESSION])

# The following keys need to match those of _metric_methods in
# _fairlearn_dashboard.py
# Issue 269 is about unifying the two sets
ACCURACY_SCORE = "accuracy_score"
BALANCED_ROOT_MEAN_SQUARED_ERROR = "balanced_root_mean_squared_error"
COUNT = "count"
F1_SCORE = "f1_score"
FALLOUT_RATE = "fallout_rate"
LOG_LOSS = "log_loss"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
MEAN_OVERPREDICTION = "overprediction"
MEAN_PREDICTION = "average"
MEAN_SQUARED_ERROR = "mean_squared_error"
MEAN_UNDERPREDICTION = "underprediction"
MISS_RATE = "miss_rate"
PRECISION_SCORE = "precision_score"
R2_SCORE = "r2_score"
RECALL_SCORE = "recall_score"
ROC_AUC_SCORE = "balanced_accuracy_score"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
SELECTION_RATE = "selection_rate"
SPECIFICITY_SCORE = "specificity_score"
ZERO_ONE_LOSS = "zero_one_loss"


class _func_wrapper:
    """
    Metric function wrapper, so they return 0 rather than raising ValueError.

    This is primarily to deal with roc_auc_score()
    """

    def __init__(self, metric_function):
        self.metric_function = metric_function
        self.__name__ = f"{metric_function.__name__}_wrapped"

    def __call__(self, y_true, y_pred):
        result = 0
        try:
            result = self.metric_function(y_true, y_pred)
        except ValueError:
            msg = "Evaluation of {0} failed. Substituting 0"
            warnings.warn(msg.format(self.metric_function.__name__))
        return result


BINARY_CLASSIFICATION_METRICS = {}
BINARY_CLASSIFICATION_METRICS[ACCURACY_SCORE] = skm.accuracy_score
BINARY_CLASSIFICATION_METRICS[COUNT] = count
BINARY_CLASSIFICATION_METRICS[FALLOUT_RATE] = false_positive_rate
BINARY_CLASSIFICATION_METRICS[F1_SCORE] = skm.f1_score
BINARY_CLASSIFICATION_METRICS[MEAN_OVERPREDICTION] = _mean_overprediction
BINARY_CLASSIFICATION_METRICS[MEAN_UNDERPREDICTION] = _mean_underprediction
BINARY_CLASSIFICATION_METRICS[MISS_RATE] = false_negative_rate
BINARY_CLASSIFICATION_METRICS[PRECISION_SCORE] = skm.precision_score
BINARY_CLASSIFICATION_METRICS[RECALL_SCORE] = skm.recall_score
BINARY_CLASSIFICATION_METRICS[ROC_AUC_SCORE] = _func_wrapper(skm.roc_auc_score)
BINARY_CLASSIFICATION_METRICS[SELECTION_RATE] = selection_rate
BINARY_CLASSIFICATION_METRICS[SPECIFICITY_SCORE] = true_negative_rate

REGRESSION_METRICS = {}
REGRESSION_METRICS[COUNT] = count
REGRESSION_METRICS[MEAN_ABSOLUTE_ERROR] = skm.mean_absolute_error
REGRESSION_METRICS[MEAN_PREDICTION] = mean_prediction
REGRESSION_METRICS[MEAN_SQUARED_ERROR] = skm.mean_squared_error
REGRESSION_METRICS[ROOT_MEAN_SQUARED_ERROR] = _root_mean_squared_error
REGRESSION_METRICS[R2_SCORE] = skm.r2_score

PROBABILITY_METRICS = {}
PROBABILITY_METRICS[
    BALANCED_ROOT_MEAN_SQUARED_ERROR
] = _balanced_root_mean_squared_error
PROBABILITY_METRICS[COUNT] = count
PROBABILITY_METRICS[LOG_LOSS] = skm.log_loss
PROBABILITY_METRICS[MEAN_ABSOLUTE_ERROR] = skm.mean_absolute_error
PROBABILITY_METRICS[MEAN_OVERPREDICTION] = _mean_overprediction
PROBABILITY_METRICS[MEAN_PREDICTION] = mean_prediction
PROBABILITY_METRICS[MEAN_SQUARED_ERROR] = skm.mean_squared_error
PROBABILITY_METRICS[MEAN_UNDERPREDICTION] = _mean_underprediction
PROBABILITY_METRICS[ROC_AUC_SCORE] = skm.roc_auc_score
PROBABILITY_METRICS[ROOT_MEAN_SQUARED_ERROR] = _root_mean_squared_error


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


def _create_group_metric_set(y_true, predictions, sensitive_features, prediction_type):
    """Create a dictionary matching the Dashboard's cache."""
    result = dict()
    result[_SCHEMA] = _DASHBOARD_DICTIONARY
    result[_VERSION] = 0

    if prediction_type not in _allowed_prediction_types:
        msg_format = "prediction_type '{0}' not in {1}"
        msg = msg_format.format(
            prediction_type, sorted(list(_allowed_prediction_types))
        )
        raise ValueError(msg)

    function_dict = None
    if prediction_type == BINARY_CLASSIFICATION:
        result[_PREDICTION_TYPE] = _PREDICTION_BINARY_CLASSIFICATION
        function_dict = BINARY_CLASSIFICATION_METRICS
    elif prediction_type == REGRESSION:
        result[_PREDICTION_TYPE] = _PREDICTION_REGRESSION
        function_dict = REGRESSION_METRICS
    elif prediction_type == PROBABILITY:
        result[_PREDICTION_TYPE] = _PREDICTION_PROBABILITY
        function_dict = PROBABILITY_METRICS
    else:
        raise NotImplementedError("No support yet for {0}".format(prediction_type))

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
                gmr = MetricFrame(
                    metrics=metric_func,
                    y_true=result[_Y_TRUE],
                    y_pred=prediction,
                    sensitive_features=g[_BIN_VECTOR],
                )
                curr_dict = dict()
                if np.issubdtype(type(gmr.overall), np.integer):
                    # Ensure return value is JSON serialisable
                    # Sometimes you have to listen carefully to the quack
                    curr_dict[_GLOBAL] = int(gmr.overall)
                    curr_dict[_BINS] = [int(x) for x in list(gmr.by_group)]
                else:
                    curr_dict[_GLOBAL] = gmr.overall
                    curr_dict[_BINS] = list(gmr.by_group)
                metric_dict[metric_key] = curr_dict
            by_prediction_list.append(metric_dict)
        result[_PRECOMPUTED_METRICS].append(by_prediction_list)

    return result
