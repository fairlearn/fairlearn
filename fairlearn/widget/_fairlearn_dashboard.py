# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""Defines the fairlearn dashboard class."""

from ._fairlearn_widget import FairlearnWidget
from fairlearn.metrics import (
    true_negative_rate_group_summary,
    false_positive_rate_group_summary,
    false_negative_rate_group_summary,
    _root_mean_squared_error_group_summary,
    _balanced_root_mean_squared_error_group_summary,
    mean_prediction_group_summary,
    selection_rate_group_summary,
    _mean_overprediction_group_summary,
    _mean_underprediction_group_summary,

    accuracy_score_group_summary,
    precision_score_group_summary,
    recall_score_group_summary,
    roc_auc_score_group_summary,
    zero_one_loss_group_summary,
    mean_absolute_error_group_summary,
    mean_squared_error_group_summary,
    r2_score_group_summary,
    f1_score_group_summary,
    log_loss_group_summary,
    )

from IPython.display import display
from scipy.sparse import issparse
import copy
import numpy as np
import pandas as pd


class FairlearnDashboard(object):
    """The dashboard class, wraps the dashboard component.

    :param sensitive_features:  A matrix of feature vector examples (# examples x # features),
        these can be from the initial dataset, or reserved from training.
    :type sensitive_features: numpy.array or list[][] or pandas.DataFrame or pandas.Series
    :param y_true: The true labels or values for the provided dataset.
    :type y_true: numpy.array or list[]
    :param y_pred: Array of output predictions from models to be evaluated. Can be a single
        array of predictions, or a 2D list over multiple models. Can be a dictionary
        of named model predictions.
    :type y_pred: numpy.array or list[][] or list[] or dict {string: list[]}
    :param sensitive_feature_names: Feature names
    :type sensitive_feature_names: numpy.array or list[]
    """

    def __init__(
            self, *,
            sensitive_features,
            y_true, y_pred,
            sensitive_feature_names=None,
            locale=None):
        """Initialize the fairlearn Dashboard."""
        self._widget_instance = FairlearnWidget()
        if sensitive_features is None or y_true is None or y_pred is None:
            raise ValueError("Required parameters not provided")

        # The following mappings should match those in the GroupMetricSet
        # Issue 269 has been opened to track the work for unifying the two
        self._metric_methods = {
            "accuracy_score": {
                "model_type": ["classification"],
                "function": accuracy_score_group_summary
            },
            "balanced_accuracy_score": {
                "model_type": ["classification"],
                "function": roc_auc_score_group_summary
            },
            "precision_score": {
                "model_type": ["classification"],
                "function": precision_score_group_summary
            },
            "recall_score": {
                "model_type": ["classification"],
                "function": recall_score_group_summary
            },
            "zero_one_loss": {
                "model_type": [],
                "function": zero_one_loss_group_summary
            },
            "specificity_score": {
                "model_type": [],
                "function": true_negative_rate_group_summary
            },
            "miss_rate": {
                "model_type": [],
                "function": false_negative_rate_group_summary
            },
            "fallout_rate": {
                "model_type": [],
                "function": false_positive_rate_group_summary
            },
            "false_positive_over_total": {
                "model_type": [],
                "function": false_positive_rate_group_summary
            },
            "false_negative_over_total": {
                "model_type": [],
                "function": false_negative_rate_group_summary
            },
            "selection_rate": {
                "model_type": [],
                "function": selection_rate_group_summary
            },
            "auc": {
                "model_type": ["probability"],
                "function": roc_auc_score_group_summary
            },
            "root_mean_squared_error": {
                "model_type": ["regression", "probability"],
                "function": _root_mean_squared_error_group_summary
            },
            "balanced_root_mean_squared_error": {
                "model_type": ["probability"],
                "function": _balanced_root_mean_squared_error_group_summary
            },
            "mean_squared_error": {
                "model_type": ["regression", "probability"],
                "function": mean_squared_error_group_summary
            },
            "mean_absolute_error": {
                "model_type": ["regression", "probability"],
                "function": mean_absolute_error_group_summary
            },
            "r2_score": {
                "model_type": ["regression"],
                "function": r2_score_group_summary
            },
            "f1_score": {
                "model_type": ["classification"],
                "function": f1_score_group_summary
            },
            "log_loss": {
                "model_type": ["probability"],
                "function": log_loss_group_summary
            },
            "overprediction": {
                "model_type": [],
                "function": _mean_overprediction_group_summary
            },
            "underprediction": {
                "model_type": [],
                "function": _mean_underprediction_group_summary
            },
            "average": {
                "model_type": [],
                "function": mean_prediction_group_summary
            }
        }

        classification_methods = [method[0] for method in self._metric_methods.items()
                                  if "classification" in method[1]["model_type"]]
        regression_methods = [method[0] for method in self._metric_methods.items()
                              if "regression" in method[1]["model_type"]]
        probability_methods = [method[0] for method in self._metric_methods.items()
                               if "probability" in method[1]["model_type"]]

        dataset = self._sanitize_data_shape(sensitive_features)
        model_names = None
        if isinstance(y_pred, dict):
            model_names = []
            self._y_pred = []
            for k, v in y_pred.items():
                model_names.append(k)
                self._y_pred.append(self._convert_to_list(v))
        else:
            self._y_pred = self._convert_to_list(y_pred)
        if len(np.shape(self._y_pred)) == 1:
            self._y_pred = [self._y_pred]
        self._y_true = self._convert_to_list(y_true)

        if np.shape(self._y_true)[0] != np.shape(self._y_pred)[1]:
            raise ValueError("Predicted y does not match true y shape")

        if np.shape(self._y_true)[0] != np.shape(dataset)[0]:
            raise ValueError("Sensitive features shape does not match true y shape")

        dataArg = {
            "true_y": self._y_true,
            "predicted_ys": self._y_pred,
            "dataset": dataset,
            "classification_methods": classification_methods,
            "regression_methods": regression_methods,
            "probability_methods": probability_methods,
            "model_names": model_names
        }

        if sensitive_feature_names is not None:
            sensitive_feature_names = self._convert_to_list(sensitive_feature_names)
            if np.shape(dataset)[1] != np.shape(sensitive_feature_names)[0]:
                raise Warning("Feature names shape does not match dataset, ignoring")
            else:
                dataArg["features"] = sensitive_feature_names

        if locale is not None:
            dataArg["locale"] = locale

        self._widget_instance.value = dataArg
        self._widget_instance.observe(self._on_request, names="request")
        display(self._widget_instance)

    def _on_request(self, change):
        try:
            new = change.new
            response = copy.deepcopy(self._widget_instance.response)
            for id in new:  # noqa: A001
                try:
                    if id not in response:
                        data = new[id]
                        method = self._metric_methods.get(data["metricKey"]).get("function")
                        binVector = data["binVector"]
                        prediction = method(
                            self._y_true,
                            self._y_pred[data["modelIndex"]],
                            sensitive_features=binVector)
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
