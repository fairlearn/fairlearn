# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
from fairlearn.postprocessing._threshold_operation import ThresholdOperation
from fairlearn.postprocessing._constants import SCORE_KEY, LABEL_KEY, ATTRIBUTE_KEY


sensitive_feature_names_ex1 = ["A", "B", "C"]
sensitive_features_ex1 = [x for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
sensitive_feature_names_ex2 = ["x", "Y"]
sensitive_features_ex2 = [x for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']
labels_ex = [int(x) for x in '0110100' '0010111' '000111']
scores_ex = [int(x) for x in '0011233' '0001111' '011112']

LabelAndPrediction = namedtuple('LabelAndPrediction', 'label prediction')


class ExamplePredictor():
    def predict(self, X):
        return scores_ex


class ExampleNotPredictor():
    pass


class ExampleEstimator():
    def fit(self, X, Y):
        pass

    def predict(self, X):
        return scores_ex


class ExampleNotEstimator1():
    def fit(self, X, Y):
        pass


class ExampleNotEstimator2():
    def predict(self, X):
        pass


def _get_grouped_data_and_base_points(attribute_value):
    data = pd.DataFrame({
        ATTRIBUTE_KEY: sensitive_features_ex1,
        SCORE_KEY: scores_ex,
        LABEL_KEY: labels_ex})
    grouped_data = data.groupby(ATTRIBUTE_KEY).get_group(attribute_value) \
        .sort_values(by=SCORE_KEY, ascending=False)
    x_grid = np.linspace(0, 1, 100)

    if attribute_value == "A":
        expected_roc_points = pd.DataFrame({
            "x": [0, 0.25, 0.5, 0.5, 1],
            "y": [0, 1/3,  2/3, 1,   1],
            "operation": [ThresholdOperation('>', np.inf),
                          ThresholdOperation('<', 0.5),
                          ThresholdOperation('<', 1.5),
                          ThresholdOperation('<', 2.5),
                          ThresholdOperation('>', -np.inf)]
        })
        ignore_for_base_points = [1, 2]

    if attribute_value == "B":
        expected_roc_points = pd.DataFrame({
            "x": [0, 1/3, 1],
            "y": [0, 3/4, 1],
            "operation": [ThresholdOperation('>', np.inf),
                          ThresholdOperation('<', 0.5),
                          ThresholdOperation('>', -np.inf)]
        })
        ignore_for_base_points = []

    if attribute_value == "C":
        expected_roc_points = pd.DataFrame({
            "x": [0, 0,   2/3, 1],
            "y": [0, 1/3,  1,  1],
            "operation": [ThresholdOperation('>', np.inf),
                          ThresholdOperation('<', 0.5),
                          ThresholdOperation('<', 1.5),
                          ThresholdOperation('>', -np.inf)]
        })
        ignore_for_base_points = [0]

    return grouped_data, expected_roc_points, ignore_for_base_points, x_grid


def _get_predictions_by_attribute(adjusted_predictor, attributes, scores, labels):
    labels_and_predictions = defaultdict(list)
    for i in range(len(attributes)):
        labels_and_predictions[attributes[i]].append(
            LabelAndPrediction(labels[i],
                               adjusted_predictor([attributes[i]], [scores[i]])))
    return labels_and_predictions


def _format_as_list_of_lists(lst):
    return [[item] for item in lst]
