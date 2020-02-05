# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
import pytest
from fairlearn._input_validation import _SENSITIVE_FEATURE_COMPRESSION_SEPARATOR
from fairlearn.postprocessing._threshold_operation import ThresholdOperation
from fairlearn.postprocessing._constants import SCORE_KEY, LABEL_KEY, SENSITIVE_FEATURE_KEY

from test.unit.input_convertors import ensure_list_1d, ensure_ndarray, ensure_ndarray_2d, \
    ensure_dataframe, ensure_series


X_ex = np.stack(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                 [5, 4, 3, 2, 7, 8, 3, 4, 4, 3, 6,  5,  7,  5,  2,  1,  9,  4,  8,  0],
                 [9, 2, 4, 2, 9, 3, 1, 8, 1, 1, 2,  9,  6,  7,  4,  2,  56, 1,  2,  34]), -1)

sensitive_feature_names_ex1 = ["A", "B", "C"]
sensitive_features_ex1 = np.array([x for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']).reshape(-1, 1)

sensitive_feature_names_ex2 = ["x", "Y"]
sensitive_features_ex2 = np.array([x for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']).reshape(-1, 1)

labels_ex = np.array([int(x) for x in '0110100' '0010111' '000111'])
degenerate_labels_ex = np.array([int(x) for x in '0000000' '0000000' '000000'])
scores_ex = np.array([int(x) for x in '0011233' '0001111' '011112'])

# combine earlier examples for a new scenario with multiple sensitive features
sensitive_features_ex3 = np.hstack((sensitive_features_ex1, sensitive_features_ex2))
sensitive_feature_names_ex3 = ["A,x", "A,Y", "B,x", "B,Y", "C,Y"]

candidate_X_transforms = [ensure_ndarray, ensure_dataframe]
candidate_Y_transforms = [ensure_list_1d, ensure_ndarray, ensure_series, ensure_dataframe]
candidate_A_transforms = [ensure_list_1d, ensure_ndarray, ensure_ndarray_2d, ensure_series,
                          ensure_dataframe]

LabelAndPrediction = namedtuple('LabelAndPrediction', 'label prediction')

_data = namedtuple('_data', 'feature_names sensitive_features X y scores')


def is_invalid_sensitive_feature_transformation(sensitive_features_data,
                                                sensitive_features_transform):
    if (sensitive_features_data == sensitive_features_ex3).all() and \
            sensitive_features_transform in [ensure_list_1d, ensure_series]:
        return True


@pytest.fixture
@pytest.mark.parametrize("feature_names,features,labels,scores",
                         [
                             (sensitive_feature_names_ex1,
                              sensitive_features_ex1,
                              labels_ex,
                              scores_ex),
                             (sensitive_feature_names_ex2,
                              sensitive_features_ex2,
                              labels_ex,
                              scores_ex),
                             (sensitive_feature_names_ex3,
                              sensitive_features_ex3,
                              labels_ex,
                              scores_ex)
                         ])
def data(feature_names, features, labels, scores):
    # note that the features are identical with sensitive features in these tests
    return _data(feature_names, features, features, labels, scores)


@pytest.fixture
@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_feature_transform", candidate_A_transforms)
def data_X_y_sf(data, X_transform, y_transform, sensitive_feature_transform):
    data.X = X_transform(data.X)
    data.y = y_transform(data.y)
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
        SENSITIVE_FEATURE_KEY: sensitive_features_ex1.reshape(-1),
        SCORE_KEY: scores_ex.reshape(-1),
        LABEL_KEY: labels_ex.reshape(-1)})
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
    sensitive_features_mapped = _map_into_single_column(sensitive_features)
    for i in range(len(sensitive_features_mapped)):
        labels_and_predictions[sensitive_features_mapped[i]].append(
            LabelAndPrediction(labels[i],
                               adjusted_predictor([sensitive_features_mapped[i]], [scores[i]])))
    return labels_and_predictions


def _map_into_single_column(matrix):
    if len(np.array(matrix).shape) == 1:
        return np.array(matrix)

    return np.apply_along_axis(
        lambda row: _SENSITIVE_FEATURE_COMPRESSION_SEPARATOR.join(
            [str(row[i])
             .replace("\\", "\\\\")  # escape backslash and separator
             .replace(_SENSITIVE_FEATURE_COMPRESSION_SEPARATOR,
                      "\\" + _SENSITIVE_FEATURE_COMPRESSION_SEPARATOR)
             for i in range(len(row))]),
        axis=1,
        arr=matrix)
