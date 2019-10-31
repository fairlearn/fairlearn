# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the fairlearn dashboard class."""

from .fairlearnWidget import FairlearnWidget
from fairlearn.metrics import group_accuracy_score, group_precision_score,\
    group_recall_score, group_zero_one_loss, group_max_error, group_mean_absolute_error,\
    group_mean_squared_error, group_mean_squared_log_error, group_median_absolute_error,\
    group_specificity_score, group_miss_rate, group_fallout_rate, group_selection_rate,\
    group_balanced_root_mean_squared_error, group_mean_overprediction,\
    group_mean_underprediction, group_mean_prediction
from IPython.display import display
from scipy.sparse import issparse
import copy
import numpy as np
import pandas as pd


class FairlearnDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(
            self, *,
            sensitive_features,
            true_y, predicted_ys,
            sensitive_feature_names=None,
            is_classifier=None):

        """Initialize the fairlearn Dashboard.

        :param sensitive_features:  A matrix of feature vector examples (# examples x # features),
            these can be from the initial dataset, or reserved from training. Currently only
            categorical features are supported
        :type sensitive_features: numpy.array or list[][] or Pandas Dataframe
        :param trueY: The true labels for the provided dataset. Will overwrite any set on
            explanation object already
        :type trueY: numpy.array or list[]
        :param predicted_ys: Array of output predictions from models to be evaluated
        :type predicted_ys: numpy.array or list[][]
        :param sensitive_feature_names: Feature names
        :type sensitive_feature_names: numpy.array or list[]
        """
        self._widget_instance = FairlearnWidget()
        if sensitive_features is None or true_y is None or predicted_ys is None:
            raise ValueError("Required parameters not provided")

        self._metric_methods = {
            "accuracy_score": {
                "model_type": ["classification"],
                "function": group_accuracy_score
            },
            "precision_score": {
                "model_type": ["classification"],
                "function": group_precision_score
            },
            "recall_score": {
                "model_type": ["classification"],
                "function": group_recall_score
            },
            "zero_one_loss": {
                "model_type": [],
                "function": group_zero_one_loss
            },
            "specificity_score": {
                "model_type": [],
                "function": group_specificity_score
            },
            "miss_rate": {
                "model_type": [],
                "function": group_miss_rate
            },
            "fallout_rate": {
                "model_type": [],
                "function": group_fallout_rate
            },
            "false_positive_over_total": {
                "model_type": [],
                "function": group_fallout_rate
            },
            "false_negative_over_total": {
                "model_type": [],
                "function": group_miss_rate
            },
            "selection_rate": {
                "model_type": [],
                "function": group_selection_rate
            },
            "max_error": {
                "model_type": ["regression"],
                "function": group_max_error
            },
            "mean_absolute_error": {
                "model_type": ["regression"],
                "function": group_mean_absolute_error
            },
            "mean_squared_error": {
                "model_type": ["regression"],
                "function": group_mean_squared_error
            },
            "mean_squared_log_error": {
                "model_type": ["regression"],
                "function": group_mean_squared_log_error
            },
            "median_absolute_error": {
                "model_type": ["regression"],
                "function": group_median_absolute_error
            },
            "balanced_root_mean_squared_error": {
                "model_type": [],
                "function": group_balanced_root_mean_squared_error
            },
            "overprediction": {
                "model_type": [],
                "function": group_mean_overprediction
            },
            "underprediction": {
                "model_type": [],
                "function": group_mean_underprediction
            },
            "average": {
                "model_type": [],
                "function": group_mean_prediction
            }
        }

        classification_methods = [method[0] for method in self._metric_methods.items()
                                  if "classification" in method[1]["model_type"]]
        regression_methods = [method[0] for method in self._metric_methods.items()
                              if "regression" in method[1]["model_type"]]

        dataset = self._sanitize_data_shape(sensitive_features)
        self._predicted_ys = self._convert_to_list(predicted_ys)
        if len(np.shape(self._predicted_ys)) == 1:
            self._predicted_ys = [self._predicted_ys]
        self._true_y = self._convert_to_list(true_y)

        if np.shape(self._true_y)[0] != np.shape(self._predicted_ys)[1]:
            raise ValueError("Predicted y does not match true y shape")

        if np.shape(self._true_y)[0] != np.shape(dataset)[0]:
            raise ValueError("Sensitive features shape does not match true y shape")

        dataArg = {
            "true_y": self._true_y,
            "predicted_ys": self._predicted_ys,
            "dataset": dataset,
            "classification_methods": classification_methods,
            "regression_methods": regression_methods
        }

        if sensitive_feature_names is not None:
            sensitive_feature_names = self._convert_to_list(sensitive_feature_names)
            if np.shape(dataset)[1] != np.shape(sensitive_feature_names)[0]:
                raise Warning("Feature names shape does not match dataset, ignoring")
            else:
                dataArg["features"] = sensitive_feature_names

        if is_classifier is not None and isinstance(is_classifier, bool):
            dataArg["is_classifier"] = is_classifier

        self._widget_instance.value = dataArg
        self._widget_instance.observe(self._on_request, names="request")
        display(self._widget_instance)

    def _on_request(self, change):
        try:
            new = change.new
            response = copy.deepcopy(self._widget_instance.response)
            for id in new:
                try:
                    if id not in response:
                        data = new[id]
                        method = self._metric_methods.get(data["metricKey"]).get("function")
                        binVector = data["binVector"]
                        prediction = method(
                            self._true_y,
                            self._predicted_ys[data["modelIndex"]],
                            binVector)
                        response[id] = {
                                "global": prediction.overall,
                                "bins": prediction.by_group
                                }
                except Exception as ed:
                    response[id] = {
                        "error": ed,
                        "global": 0,
                        "bins": []}
            self._widget_instance.response = response
        except Exception:
            raise ValueError("Error while making request")

    def _show(self):
        display(self._widget_instance)

    def _sanitize_data_shape(self, dataset):
        result = self._convert_to_list(dataset)
        # Dataset should be 2d, if not we need to map
        if (len(np.shape(result)) == 2):
            return result
        return list(map(lambda x: [x], result))

    def _convert_to_list(self, array):
        if issparse(array):
            if array.shape[1] > 1000:
                raise ValueError("Exceeds maximum number of features for visualization (1000)")
            return array.toarray().tolist()

        if (isinstance(array, pd.DataFrame) or isinstance(array, pd.Series)):
            return array.values.tolist()
        if (isinstance(array, np.ndarray)):
            return array.tolist()
        return array
