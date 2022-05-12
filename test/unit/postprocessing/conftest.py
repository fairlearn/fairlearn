# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from collections import defaultdict, namedtuple
from test.unit.input_convertors import (
    _map_into_single_column,
    ensure_dataframe,
    ensure_list_1d,
    ensure_ndarray,
    ensure_ndarray_2d,
    ensure_series,
)

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from fairlearn.postprocessing._constants import (
    LABEL_KEY,
    SCORE_KEY,
    SENSITIVE_FEATURE_KEY,
)
from fairlearn.postprocessing._threshold_operation import ThresholdOperation

X_ex = np.stack(
    (
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [5, 4, 3, 2, 7, 8, 3, 4, 4, 3, 6, 5, 7, 5, 2, 1, 9, 4, 8, 0],
        [9, 2, 4, 2, 9, 3, 1, 8, 1, 1, 2, 9, 6, 7, 4, 2, 56, 1, 2, 34],
    ),
    -1,
)

sensitive_feature_names_ex1 = ["A", "B", "C"]
sensitive_features_ex1 = np.array([x for x in "AAAAAAABBBBBBBCCCCCC"]).reshape(-1, 1)

sensitive_feature_names_ex2 = ["x", "Y"]
sensitive_features_ex2 = np.array([x for x in "xxxYYYYxYYYYYxYYYYYY"]).reshape(-1, 1)

labels_ex = np.array([int(x) for x in "01101000010111000111"])
degenerate_labels_ex = np.array([int(x) for x in "00000000000000000000"])
scores_ex = np.array([int(x) for x in "00112330001111011112"])

# combine earlier examples for a new scenario with multiple sensitive features
sensitive_features_ex3 = np.hstack((sensitive_features_ex1, sensitive_features_ex2))
sensitive_feature_names_ex3 = ["A,x", "A,Y", "B,x", "B,Y", "C,Y"]

candidate_X_transforms = [ensure_ndarray, ensure_dataframe]
candidate_Y_transforms = [
    ensure_list_1d,
    ensure_ndarray,
    ensure_series,
    ensure_dataframe,
]
candidate_A_transforms = [
    ensure_list_1d,
    ensure_ndarray,
    ensure_ndarray_2d,
    ensure_series,
    ensure_dataframe,
]

LabelAndPrediction = namedtuple("LabelAndPrediction", "label prediction")

_data = namedtuple("_data", "example_name feature_names sensitive_features X y scores")

_data_ex1 = _data(
    "example 1",
    sensitive_feature_names_ex1,
    sensitive_features_ex1,
    X_ex,
    labels_ex,
    scores_ex,
)

_data_ex2 = _data(
    "example 2",
    sensitive_feature_names_ex2,
    sensitive_features_ex2,
    X_ex,
    labels_ex,
    scores_ex,
)

_data_ex3 = _data(
    "example 3",
    sensitive_feature_names_ex3,
    sensitive_features_ex3,
    X_ex,
    labels_ex,
    scores_ex,
)


@pytest.fixture(params=[_data_ex1, _data_ex2, _data_ex3])
def data(request):
    return request.param


def is_invalid_transformation(**kwargs):
    sensitive_feature_transform = kwargs["data_sf"]
    sensitive_features = kwargs["data"].sensitive_features

    # Skip combinations where the multi-column sensitive features would have to be compressed
    # into a one-dimensional data structure.
    if (
        sensitive_features == sensitive_features_ex3
    ).all() and sensitive_feature_transform in [ensure_list_1d, ensure_series]:
        return True
    return False


@pytest.fixture(params=candidate_A_transforms)
def data_sf(data, request):  # sf is an abbreviation for sensitive features
    sensitive_feature_transform = request.param
    data._replace(
        sensitive_features=sensitive_feature_transform(data.sensitive_features)
    )
    return data


@pytest.fixture(params=candidate_X_transforms)
def data_X_sf(data_sf, request):  # sf is an abbreviation for sensitive features
    X_transform = request.param
    data_sf._replace(X=X_transform(data_sf.X))
    return data_sf


@pytest.fixture(params=candidate_Y_transforms)
def data_X_y_sf(data_X_sf, request):  # sf is an abbreviation for sensitive features
    y_transform = request.param
    data_X_sf._replace(y=y_transform(data_X_sf.y))
    return data_X_sf


class ExamplePredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, scores):
        self.scores = scores

    def fit(self, X, y=None, **kwargs):
        self.scores_ = self.scores
        return self

    def predict(self, X):
        return self.scores_


class ExampleNotPredictor:
    pass


class ExampleNotEstimator1:
    def fit(self, X, Y):
        pass


class ExampleNotEstimator2:
    def predict(self, X):
        pass


def _get_grouped_data_and_base_points(sensitive_feature_value):
    data = pd.DataFrame(
        {
            SENSITIVE_FEATURE_KEY: sensitive_features_ex1.squeeze(),
            SCORE_KEY: scores_ex.squeeze(),
            LABEL_KEY: labels_ex.squeeze(),
        }
    )
    grouped_data = (
        data.groupby(SENSITIVE_FEATURE_KEY)
        .get_group(sensitive_feature_value)
        .sort_values(by=SCORE_KEY, ascending=False)
    )
    x_grid = np.linspace(0, 1, 100)

    if sensitive_feature_value == "A":
        expected_roc_points = pd.DataFrame(
            {
                "x": [0, 0.25, 0.5, 0.5, 1],
                "y": [0, 1 / 3, 2 / 3, 1, 1],
                "operation": [
                    ThresholdOperation(">", np.inf),
                    ThresholdOperation("<", 0.5),
                    ThresholdOperation("<", 1.5),
                    ThresholdOperation("<", 2.5),
                    ThresholdOperation(">", -np.inf),
                ],
            }
        )
        ignore_for_base_points = [1, 2]

    if sensitive_feature_value == "B":
        expected_roc_points = pd.DataFrame(
            {
                "x": [0, 1 / 3, 1],
                "y": [0, 3 / 4, 1],
                "operation": [
                    ThresholdOperation(">", np.inf),
                    ThresholdOperation("<", 0.5),
                    ThresholdOperation(">", -np.inf),
                ],
            }
        )
        ignore_for_base_points = []

    if sensitive_feature_value == "C":
        expected_roc_points = pd.DataFrame(
            {
                "x": [0, 0, 2 / 3, 1],
                "y": [0, 1 / 3, 1, 1],
                "operation": [
                    ThresholdOperation(">", np.inf),
                    ThresholdOperation("<", 0.5),
                    ThresholdOperation("<", 1.5),
                    ThresholdOperation(">", -np.inf),
                ],
            }
        )
        ignore_for_base_points = [0]

    return grouped_data, expected_roc_points, ignore_for_base_points, x_grid


def _get_predictions_by_sensitive_feature(
    adjusted_predictor, sensitive_features, scores, labels
):
    labels_and_predictions = defaultdict(list)
    sensitive_features_mapped = _map_into_single_column(sensitive_features)
    for i in range(len(sensitive_features_mapped)):
        labels_and_predictions[sensitive_features_mapped[i]].append(
            LabelAndPrediction(
                labels[i],
                adjusted_predictor([sensitive_features_mapped[i]], [scores[i]]),
            )
        )
    return labels_and_predictions
