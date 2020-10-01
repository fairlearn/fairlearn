# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from fairlearn.postprocessing._threshold_operation import ThresholdOperation
from fairlearn.postprocessing._tradeoff_curve_utilities import (
    _calculate_tradeoff_points,
    _filter_points_to_get_convex_hull,
    _tradeoff_curve,
    _interpolate_curve)
from fairlearn.postprocessing._constants import SCORE_KEY, LABEL_KEY, SENSITIVE_FEATURE_KEY
from .conftest import (
    sensitive_features_ex1, labels_ex, scores_ex,
    _get_grouped_data_and_base_points, sensitive_feature_names_ex1)


def test_assert_interpolated_curve():
    # An easily interpretable test to make sure the assertion method works as expected
    base_points = pd.DataFrame({
        "x":         [0, 5, 10],
        "y":         [0, 2.5, 5],
        "operation": ["a", "b", "c"]  # irrelevant
    })
    x_grid = np.linspace(0, 10, 333)
    curve = _interpolate_curve(base_points, "x", "y", "operation", x_grid)

    _assert_interpolated_points_are_between_base_points(base_points, curve)


def test_interpolate_curve():
    # The operation is irrelevant in this case since its semantics are not
    # used within _interpolate_curve.
    base_points = pd.DataFrame({
        "x":         [0,    1,   2,   3,   4,   5,   6,   7,   8,   9],
        "y":         [-5,  -2, -1.5, -1,   0,  0.5, 0.8, 1.0, 1.1, 1.15],
        "operation": ["i", "r", "r", "e", "l", "e", "v", "a", "n", "t"]
    })
    x_grid = np.linspace(0, 9, 100)
    curve = _interpolate_curve(base_points, "x", "y", "operation", x_grid)

    _assert_interpolated_points_are_between_base_points(base_points, curve)


@pytest.mark.parametrize("base_points,expected_remaining_indices",
                         [(pd.DataFrame({
                             "x":         [0, 1],
                             "y":         [0, 1],
                             "operation": ["i", "r"]
                         }), [0, 1]),
                             (pd.DataFrame({
                              "x":         [0, 0.5, 1],
                              "y":         [0, 0.5, 1],
                              "operation": ["i", "r", "e"]
                              }), [0, 2]),
                             (pd.DataFrame({
                              "x":         [0, 0.5, 1],
                              "y":         [0, 0.51, 1],
                              "operation": ["i", "r", "e"]
                              }), [0, 1, 2]),
                             (pd.DataFrame({
                              "x":         [0, 0.2, 0.3,  0.5, 1],
                              "y":         [0, 0.3, 0.35, 0.7, 1],
                              "operation": ["i", "r", "r", "e", "l"]
                              }), [0, 1, 3, 4]),
                             (pd.DataFrame({
                              "x":         [0, 0.1, 0.2, 0.5, 1],
                              "y":         [0, 0.3, 0.5, 0.9, 1],
                              "operation": ["i", "r", "r", "e", "l"]
                              }), [0, 1, 2, 3, 4]),
                             (pd.DataFrame({
                              "x":         [0, 0.2, 0.3,  0.5, 1],
                              "y":         [0, 0.3, 0.8, 0.82, 1],
                              "operation": ["i", "r", "r", "e", "l"]
                              }), [0, 2, 4])
                         ])
def test_convex_hull(base_points, expected_remaining_indices):
    convex_hull = _filter_points_to_get_convex_hull(base_points)
    assert (base_points.x[expected_remaining_indices] == [point.x for point in convex_hull]).all()
    assert (base_points.y[expected_remaining_indices] == [point.y for point in convex_hull]).all()
    assert (base_points.operation[expected_remaining_indices] ==
            [point.operation for point in convex_hull]).all()


def test_calculate_tradeoff_points():
    data = pd.DataFrame({
        SENSITIVE_FEATURE_KEY: sensitive_features_ex1.squeeze(),
        SCORE_KEY: scores_ex.squeeze(),
        LABEL_KEY: labels_ex.squeeze()})
    grouped_data = data.groupby(SENSITIVE_FEATURE_KEY).get_group("A") \
        .sort_values(by=SCORE_KEY, ascending=False)

    roc_points = _calculate_tradeoff_points(grouped_data, "A", flip=True)
    expected_roc_points = pd.DataFrame({
        "x": [0, .0, 0.25, .5, .5,    0.5, 0.5, .75,    1.0, 1],
        "y": [0, .0, 1/3,  .0, .1/.3, 2/3, 1,   .2/.3,  1.0, 1],
        "operation": [ThresholdOperation('>', np.inf),
                      ThresholdOperation('<', -np.inf),
                      ThresholdOperation('<', 0.5),
                      ThresholdOperation('>', 2.5),
                      ThresholdOperation('>', 1.5),
                      ThresholdOperation('<', 1.5),
                      ThresholdOperation('<', 2.5),
                      ThresholdOperation('>', 0.5),
                      ThresholdOperation('<', np.inf),
                      ThresholdOperation('>', -np.inf)]
    })

    _assert_equal_points(expected_roc_points, roc_points)

    expected_roc_convex_hull = pd.DataFrame({
        "x": [0, 0.5, 1],
        "y": [0, 1,   1],
        "operation": [ThresholdOperation('>', np.inf),
                      ThresholdOperation('<', 2.5),
                      ThresholdOperation('>', -np.inf),
                      ]
    })
    # Try filtering to get the convex hull of the ROC points.
    selected_points = \
        pd.DataFrame(_filter_points_to_get_convex_hull(roc_points))[['x', 'y', 'operation']]
    _assert_equal_points(expected_roc_convex_hull, selected_points)


def test_tradeoff_curve():
    for sensitive_feature_value in sensitive_feature_names_ex1:
        grouped_data, base_points, ignore_for_base_points, x_grid = \
            _get_grouped_data_and_base_points(sensitive_feature_value)

        roc_convex_hull = _tradeoff_curve(grouped_data, x_grid, sensitive_feature_value)
        curve = _interpolate_curve(roc_convex_hull, 'x', 'y', 'operation', x_grid)

        _assert_interpolated_points_are_between_base_points(base_points, curve,
                                                            ignore_for_base_points)


def _assert_interpolated_points_are_between_base_points(base_points, curve,
                                                        ignore_for_base_points=None):
    def _get_base_point_coordinates(i, data):
        return data.x[i], data.y[i]

    if ignore_for_base_points is None:
        ignore_for_base_points = []

    # Determine base point indices from base points such that points which are
    # not corners on the convex hull are ignored. These points are listed by
    # index in ignore_for_base_points.
    start_base_point_index = 0
    while start_base_point_index in ignore_for_base_points:
        start_base_point_index += 1

    base_point_index = start_base_point_index + 1
    while base_point_index in ignore_for_base_points:
        base_point_index += 1

    current_base_point_x, current_base_point_y = \
        _get_base_point_coordinates(start_base_point_index, base_points)
    next_base_point_x, next_base_point_y = \
        _get_base_point_coordinates(base_point_index, base_points)

    for x_grid_index in range(len(curve)):
        x = curve.x[x_grid_index]
        y = curve.y[x_grid_index]
        if np.isclose(x, current_base_point_x):
            assert np.isclose(y, current_base_point_y)
            continue

        while x > next_base_point_x:
            current_base_point_x, current_base_point_y = \
                _get_base_point_coordinates(base_point_index, base_points)
            base_point_index += 1
            while base_point_index in ignore_for_base_points:
                base_point_index += 1
            next_base_point_x, next_base_point_y = \
                _get_base_point_coordinates(base_point_index, base_points)

        if np.isclose(x, current_base_point_x):
            assert np.isclose(y, current_base_point_y)
            continue

        if np.isclose(x, next_base_point_x):
            assert np.isclose(y, next_base_point_y)
            continue

        # We know that current_base_point_x < x < next_base_point_x.
        # Ensure that the curve point lies exactly between the two base points
        # by checking the slope of the lines connecting the curve point to the
        # base points.
        assert np.isclose((y - current_base_point_y) / (x - current_base_point_x),
                          (next_base_point_y - y) / (next_base_point_x - x))


def _assert_equal_points(expected_points, actual_points):
    assert len(expected_points) == len(actual_points)

    for i in range(len(expected_points)):
        assert np.isclose(actual_points.x[i], expected_points.x[i])
        assert np.isclose(actual_points.y[i], expected_points.y[i])
        assert actual_points.operation[i].operator == \
            expected_points.operation[i].operator
        assert np.isclose(actual_points.operation[i].threshold,
                          expected_points.operation[i].threshold)
