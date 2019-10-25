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
import numpy as np
import pandas as pd


class FairlearnDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(
            self, *,
            sensitive_features,
            true_y, predicted_ys,
            class_names=None,
            feature_names=None,
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
        :param class_names: The class names
        :type class_names: numpy.array or list[]
        :param feature_names: Feature names
        :type feature_names: numpy.array or list[]
        """
        self._widget_instance = FairlearnWidget()
        if sensitive_features is None or true_y is None or predicted_ys is None:
            raise ValueError("Required parameters not provided")
        self._true_y = true_y
        self._predicted_ys = predicted_ys

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
                "model_type": ["classification"],
                "function": group_zero_one_loss
            },
            "specificity_score": {
                "model_type": ["classification"],
                "function": group_specificity_score
            },
            "miss_rate": {
                "model_type": ["classification"],
                "function": group_miss_rate
            },
            "fallout_rate": {
                "model_type": ["classification"],
                "function": group_fallout_rate
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
            },
        }

        classification_methods = [method[0] for method in self._metric_methods.items()
                                  if "classification" in method[1]["model_type"]]
        regression_methods = [method[0] for method in self._metric_methods.items()
                              if "regression" in method[1]["model_type"]]

        dataArg = {
            "true_y": self._convertToList(true_y),
            "predicted_ys": self._convertToList(predicted_ys),
            "dataset": self._convertToList(sensitive_features),
            "classification_methods": classification_methods,
            "regression_methods": regression_methods
        }

        if feature_names is not None:
            dataArg["features"] = self._convertToList(feature_names)

        if class_names is not None:
            dataArg["classes"] = self._convertToList(class_names)

        if is_classifier is not None and isinstance(is_classifier, bool):
            dataArg["is_classifier"] = is_classifier

        self._widget_instance.value = dataArg
        self._widget_instance.observe(self._on_request, names="request")
        display(self._widget_instance)

    def _on_request(self, change):
        try:
            data = change.new["data"]
            method = self._metric_methods.get(data["metricKey"]).get("function")
            binVector = data["binVector"]
            prediction = method(self._true_y, self._predicted_ys[data["modelIndex"]], binVector)
            self._widget_instance.response = {
                "data": {
                    "global": prediction.overall,
                    "bins": prediction.by_group
                    },
                "id": change.new["id"]}
        except Exception as ed:
            self._widget_instance.response = {
                "error": ed,
                "data": {},
                "id": change.new["id"]}

    def _show(self):
        display(self._widget_instance)

    def _convertToList(self, array):
        if issparse(array):
            if array.shape[1] > 1000:
                raise ValueError("Exceeds maximum number of features for visualization (1000)")
            return array.toarray().tolist()
        if (isinstance(array, pd.DataFrame)):
            return array.values.tolist()
        if (isinstance(array, np.ndarray)):
            return array.tolist()
        return array
