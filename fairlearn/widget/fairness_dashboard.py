# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Defines the fairness dashboard class."""

from rai_core_flask import FlaskHelper
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

from flask import jsonify, request
from IPython.display import display, HTML
from jinja2 import Environment, PackageLoader
import json
import numpy as np
import os
import pandas as pd
from scipy.sparse import issparse


class FairlearnDashboard(object):
    """The dashboard class, wraps the dashboard component.

    :param sensitive_features: A matrix of feature vector examples (# examples x # features),
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

    env = Environment(loader=PackageLoader(__name__, 'templates'))
    default_template = env.get_template("inlineDashboard.html")
    _dashboard_js = None
    fairness_inputs = {}
    model_count = 0
    _service = None

    # The following mappings should match those in the GroupMetricSet
    # Issue 269 has been opened to track the work for unifying the two
    _metric_methods = {
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

    classification_methods = [method[0] for method in _metric_methods.items()
                              if "classification" in method[1]["model_type"]]
    regression_methods = [method[0] for method in _metric_methods.items()
                          if "regression" in method[1]["model_type"]]
    probability_methods = [method[0] for method in _metric_methods.items()
                           if "probability" in method[1]["model_type"]]

    @FlaskHelper.app.route('/')
    def list_view():  # noqa: D102
        return "No global list view supported at this time."

    @FlaskHelper.app.route('/<id>')
    def fairness_visual_view(id):  # noqa: D102, A002
        if id in FairlearnDashboard.fairness_inputs:
            return generate_inline_html(FairlearnDashboard.fairness_inputs[id], None)
        else:
            return "Unknown model id."

    @FlaskHelper.app.route('/<id>/metrics', methods=['POST'])
    def fairness_metrics_calc(id):  # noqa: D102, A002
        try:
            data = request.get_json(force=True)
            if id in FairlearnDashboard.fairness_inputs:
                data.update(FairlearnDashboard.fairness_inputs[id])

                if type(data["binVector"][0]) == np.int32:
                    data['binVector'] = [str(bin_) for bin_ in data['binVector']]

                method = FairlearnDashboard._metric_methods \
                    .get(data["metricKey"]).get("function")
                prediction = method(
                    data['true_y'],
                    data['predicted_ys'][data["modelIndex"]],
                    sensitive_features=data["binVector"])
                return jsonify({"data": {
                    "global": prediction.overall,
                    "bins": list(prediction.by_group.values())
                }})
        except Exception as ex:
            # debug only
            # import sys
            # import traceback
            # exc_type, exc_value, exc_traceback = sys.exc_info()

            return jsonify({
                "error": str(ex),
                # "stacktrace": str(repr(traceback.format_exception(
                #     exc_type, exc_value, exc_traceback))),
                # "locals": str(locals()),
            })

    def __init__(
            self, *,
            sensitive_features,
            y_true,
            y_pred,
            sensitive_feature_names=None,
            locale=None,
            port=None):
        """Initialize the fairness dashboard."""
        if sensitive_features is None or y_true is None or y_pred is None:
            raise ValueError("Required parameters not provided")

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
            raise ValueError("Sensitive features shape does not match true y "
                             "shape")

        fairness_input = {
            "true_y": self._y_true,
            "predicted_ys": self._y_pred,
            "dataset": dataset,
            "classification_methods": FairlearnDashboard.classification_methods,
            "regression_methods": FairlearnDashboard.regression_methods,
            "probability_methods": FairlearnDashboard.probability_methods,
        }

        if model_names is not None:
            fairness_input['model_names'] = model_names

        if locale is not None:
            fairness_input['locale'] = locale

        if sensitive_feature_names is not None:
            sensitive_feature_names = self._convert_to_list(
                sensitive_feature_names)
            if np.shape(dataset)[1] != np.shape(sensitive_feature_names)[0]:
                raise Warning("Feature names shape does not match dataset, "
                              "ignoring")
            fairness_input["features"] = sensitive_feature_names

        self._load_local_js()

        if FairlearnDashboard._service is None:
            try:
                FairlearnDashboard._service = FlaskHelper(port=port)
            except Exception as e:
                FairlearnDashboard._service = None
                raise e

        FairlearnDashboard.model_count += 1
        model_count = FairlearnDashboard.model_count

        local_url = f"{FairlearnDashboard._service.env.base_url}/{model_count}"
        metrics_url = f"{local_url}/metrics"

        fairness_input['metricsUrl'] = metrics_url

        # TODO
        fairness_input['withCredentials'] = False

        FairlearnDashboard.fairness_inputs[str(model_count)] = fairness_input

        html = generate_inline_html(fairness_input, local_url)
        # TODO
        # FairlearnDashboard._service.env.display(html)
        display(HTML(html))

    def _load_local_js(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        js_path = os.path.join(script_path, "static", "index.js")
        with open(js_path, "r", encoding="utf-8") as f:
            FairlearnDashboard._dashboard_js = f.read()

    def _sanitize_data_shape(self, dataset):
        result = self._convert_to_list(dataset)
        # Dataset should be 2d, if not we need to map
        if (len(np.shape(result)) == 2):
            return result
        return list(map(lambda x: [x], result))

    def _convert_to_list(self, array):
        if issparse(array):
            if array.shape[1] > 1000:
                raise ValueError("Exceeds maximum number of features for "
                                 "visualization (1000)")
            return array.toarray().tolist()

        if (isinstance(array, pd.DataFrame) or isinstance(array, pd.Series)):
            return array.values.tolist()
        if (isinstance(array, np.ndarray)):
            return array.tolist()
        return array


def generate_inline_html(fairness_input, local_url):  # noqa: D102, D103
    return FairlearnDashboard.default_template.render(
        fairness_input=json.dumps(fairness_input),
        main_js=FairlearnDashboard._dashboard_js,
        app_id='app_fairness',
        local_url=local_url,
        has_local_url=local_url is not None)
