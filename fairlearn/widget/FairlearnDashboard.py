# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Fairlearn dashboard class."""

from .FairlearnWidget import FairlearnWidget
from fairlearn.metrics import group_accuracy_score, group_precision_score, group_recall_score, group_zero_one_loss
from fairlearn.metrics import group_max_error, group_mean_absolute_error, group_mean_squared_error, group_mean_squared_log_error, group_median_absolute_error
from IPython.display import display
from scipy.sparse import issparse
import numpy as np
import pandas as pd



class FairlearnDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(self, *, augmented_dataset, true_y, predicted_ys, class_names=None, feature_names=None, is_classifier=None):
        """Initialize the Explanation Dashboard.

        :param explanationObject: An object that represents an explanation.
        :type explanationObject: ExplanationMixin
        :param model: An object that represents a model. It is assumed that for the classification case
            it has a method of predict_proba() returning the prediction probabilities for each
            class and for the regression case a method of predict() returning the prediction value.
        :type model: object
        :param datasetX:  A matrix of feature vector examples (# examples x # features), the same samples
            used to build the explanationObject. Will overwrite any set on explanation object already
        :type datasetX: numpy.array or list[][]
        :param trueY: The true labels for the provided dataset. Will overwrite any set on
            explanation object already
        :type trueY: numpy.array or list[]
        :param classes: The class names
        :type classes: numpy.array or list[]
        :param features: Feature names
        :type features: numpy.array or list[]
        """
        self._widget_instance = FairlearnWidget()
        if augmented_dataset is None or true_y is None or predicted_ys is None:
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
            }
        }

        classification_methods = [method[0] for method in self._metric_methods.items() if "classification" in method[1]["model_type"]]
        regression_methods = [method[0] for method in self._metric_methods.items() if "regression" in method[1]["model_type"]]

        dataArg = {
            "true_y": self._convertToList(true_y),
            "predicted_ys": self._convertToList(predicted_ys),
            "dataset": self._convertToList(augmented_dataset),
            "classification_methods": classification_methods,
            "regression_methods": regression_methods
        }

        if feature_names is not None:
            dataArg["features"] = feature_names
        
        if class_names is not None:
            dataArg["classes"] = class_names

        if is_classifier is not None:
            dataArg["is_classifier"] = is_classifier

        self._widget_instance.value = dataArg
        self._widget_instance.observe(self._on_request, names="request")
        display(self._widget_instance)

    def _on_request(self, change):
        try:
            data = change.new["data"]
            method = self._metric_methods.get(data[0])
            prediction = method(self._true_y, self._predicted_ys[data[2], data[1])
            self._widget_instance.response = {
                "data": {
                    "global": prediction.overall,
                    "bins": prediction.by_group
                    },
                "id": change.new["id"]}
        except Exception:
            self._widget_instance.response = {
                "error": "Model threw exeption while predicting",
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
