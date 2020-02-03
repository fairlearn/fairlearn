# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import pytest
from fairlearn._input_validation import \
    (_ALLOWED_INPUT_TYPES_X,
     _ALLOWED_INPUT_TYPES_Y,
     _ALLOWED_INPUT_TYPES_SENSITIVE_FEATURES)
from fairlearn.postprocessing._threshold_operation import ThresholdOperation
from fairlearn.postprocessing._constants import SCORE_KEY, LABEL_KEY, SENSITIVE_FEATURE_KEY


sensitive_feature_names_ex1 = ["A", "B", "C"]
sensitive_features_ex1_flat = [x for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
sensitive_features_ex1 = [[x] for x in sensitive_features_ex1_flat]
sensitive_feature_names_ex2 = ["x", "Y"]
sensitive_features_ex2_flat = [x for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']
sensitive_features_ex2 = [[x] for x in sensitive_features_ex2_flat]
labels_ex = [int(x) for x in '0110100' '0010111' '000111']
degenerate_labels_ex = [int(x) for x in '0000000' '0000000' '000000']
scores_ex = [int(x) for x in '0011233' '0001111' '011112']
# combine earlier examples for a new scenario with multiple sensitive features
sensitive_features_ex3 = [[sensitive_features_ex1[i], sensitive_features_ex2[i]]
                          for i in range(len(sensitive_features_ex1))]

ALLOWED_TYPES_X_CONSTRUCTORS = [np.array if f == np.ndarray else f for f in _ALLOWED_INPUT_TYPES_X]  # noqa: E501
ALLOWED_TYPES_Y_CONSTRUCTORS = [np.array if f == np.ndarray else f for f in _ALLOWED_INPUT_TYPES_Y]  # noqa: E501
ALLOWED_TYPES_SENSITIVE_FEATURES_CONSTRUCTORS = [np.array if f == np.ndarray else f for f in _ALLOWED_INPUT_TYPES_SENSITIVE_FEATURES]  # noqa: E501



LabelAndPrediction = namedtuple('LabelAndPrediction', 'label prediction')


_data = namedtuple('_data', 'feature_names sensitive_features X labels scores')

@pytest.fixture
@pytest.mark.parametrize("feature_names,flat_data,labels,scores",
                         [
                             (sensitive_feature_names_ex1,
                              sensitive_features_ex1_flat,
                              labels_ex,
                              scores_ex),
                             (sensitive_feature_names_ex2,
                              sensitive_features_ex2_flat,
                              labels_ex,
                              scores_ex)
                         ])
def data(feature_names, flat_data, labels, scores):
    feature_data = [[x] for x in flat_data]
    return _data(feature_names, flat_data, feature_data, labels, scores)


@pytest.fixture
@pytest.mark.parametrize("X_transform", ALLOWED_TYPES_X_CONSTRUCTORS)
@pytest.mark.parametrize("y_transform", ALLOWED_TYPES_Y_CONSTRUCTORS)
@pytest.mark.parametrize("sensitive_feature_transform", ALLOWED_TYPES_Y_CONSTRUCTORS)
def data_X_y_sf(data, X_transform, y_transform, sensitive_feature_transform):
    data.X = X_transform(data.X)
    data.labels = y_transform(data.labels)
    data.sensitive_features = sensitive_feature_transform(data.sensitive_features)
    return data


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


def _get_grouped_data_and_base_points(sensitive_feature_value):
    data = pd.DataFrame({
        SENSITIVE_FEATURE_KEY: sensitive_features_ex1_flat,
        SCORE_KEY: scores_ex,
        LABEL_KEY: labels_ex})
    grouped_data = data.groupby(SENSITIVE_FEATURE_KEY).get_group(sensitive_feature_value) \
        .sort_values(by=SCORE_KEY, ascending=False)
    x_grid = np.linspace(0, 1, 100)

    if sensitive_feature_value == "A":
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

    if sensitive_feature_value == "B":
        expected_roc_points = pd.DataFrame({
            "x": [0, 1/3, 1],
            "y": [0, 3/4, 1],
            "operation": [ThresholdOperation('>', np.inf),
                          ThresholdOperation('<', 0.5),
                          ThresholdOperation('>', -np.inf)]
        })
        ignore_for_base_points = []

    if sensitive_feature_value == "C":
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


def _get_predictions_by_sensitive_feature(adjusted_predictor, sensitive_features, scores, labels):
    labels_and_predictions = defaultdict(list)
    for i in range(len(sensitive_features)):
        labels_and_predictions[sensitive_features[i]].append(
            LabelAndPrediction(labels[i],
                               adjusted_predictor([sensitive_features[i]], [scores[i]])))
    return labels_and_predictions
