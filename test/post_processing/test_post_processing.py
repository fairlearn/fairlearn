# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict, namedtuple
import copy
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fairlearn.post_processing.threshold_operation import ThresholdOperation
from fairlearn.post_processing.roc_curve_based_post_processing import (roc_curve_based_post_processing_demographic_parity,
                                                                       roc_curve_based_post_processing_equalized_odds,
                                                                       _interpolate_curve,
                                                                       _get_roc,
                                                                       _calculate_roc_points,
                                                                       _filter_points_to_get_convex_hull,
                                                                       SCORE_KEY,
                                                                       LABEL_KEY,
                                                                       ATTRIBUTE_KEY,
                                                                       DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE,
                                                                       EMPTY_INPUT_ERROR_MESSAGE,
                                                                       NON_BINARY_LABELS_ERROR_MESSAGE)

example_attribute_names1 = ["A", "B", "C"]
example_attributes1 = [x for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
example_attributes2 = [x for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']
example_labels = [int(x) for x in '0110100' '0010111' '000111']
example_scores = [int(x) for x in '0011233' '0001111' '011112']

LabelAndPrediction = namedtuple('LabelAndPrediction', 'label prediction')

def test_predict_from_operation_less():
    classifier = ThresholdOperation('<', 0.5).get_predictor_from_operation()
    assert classifier(-10000) == 1
    assert classifier(0) == 1
    assert classifier(0.5) == 0
    assert classifier(1) == 0
    assert classifier(10000) == 0

def test_predict_from_operation_more():
    classifier = ThresholdOperation('>', 0.5).get_predictor_from_operation()
    assert classifier(-10000) == 0
    assert classifier(0) == 0
    assert classifier(0.5) == 0
    assert classifier(1) == 1
    assert classifier(10000) == 1

def test_predict_from_operation_invalid_operator():
    with pytest.raises(ValueError, match="Unrecognized operator: ="):
        ThresholdOperation('=', 0.5)

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
        "x":         [ 0,   1,   2,   3,   4,   5,   6,   7,   8,   9],
        "y":         [-5,  -2, -1.5, -1,   0,  0.5, 0.8, 1.0, 1.1, 1.15],
        "operation": ["i", "r", "r", "e", "l", "e", "v", "a", "n", "t"]
    })
    x_grid = np.linspace(0, 9, 100)
    curve = _interpolate_curve(base_points, "x", "y", "operation", x_grid)

    _assert_interpolated_points_are_between_base_points(base_points, curve)

def test_convex_hull():
    # Point (0.3, 0.35) lies below the line between the adjacent points
    # and therefore should be dropped for convex hull.
    base_points = pd.DataFrame({
        "x":         [0, 0.2, 0.3,  0.5, 1],
        "y":         [0, 0.3, 0.35, 0.7, 1],
        "operation": ["i", "r", "r", "e", "l"]
    })
    convex_hull = _filter_points_to_get_convex_hull(base_points)
    expected_remaining_indices = [0, 1, 3, 4]
    print([point.x for point in convex_hull])
    assert (base_points.x[expected_remaining_indices] == [point.x for point in convex_hull]).all()
    assert (base_points.y[expected_remaining_indices] == [point.y for point in convex_hull]).all()
    assert (base_points.operation[expected_remaining_indices] == [point.operation for point in convex_hull]).all()

def test_calculate_roc_points():
    data = pd.DataFrame({ATTRIBUTE_KEY: example_attributes1, SCORE_KEY: example_scores, LABEL_KEY: example_labels})
    grouped_data = data.groupby(ATTRIBUTE_KEY).get_group("A").sort_values(by=SCORE_KEY, ascending=False)

    roc_points = _calculate_roc_points(grouped_data, "A")
    expected_roc_points = pd.DataFrame({
        "x": [0, 0.25, 0.5, 0.5, 1],
        "y": [0, 1/3,  2/3, 1,   1],
        "operation": [ThresholdOperation('>', np.inf),
                      ThresholdOperation('<', 0.5),
                      ThresholdOperation('<', 1.5),
                      ThresholdOperation('<', 2.5),
                      ThresholdOperation('>', -np.inf)]
    })

    _assert_equal_points(expected_roc_points, roc_points)

    # Try filtering to get the convex hull of the ROC points.
    # This should drop the second and third point.
    selected_points = pd.DataFrame(_filter_points_to_get_convex_hull(roc_points))[['x', 'y', 'operation']]
    _assert_equal_points(expected_roc_points, selected_points, ignore_indices=[1, 2])

def test_get_roc():
    for attribute_value in ['A', 'B', 'C']:
        grouped_data, base_points, ignore_for_base_points, x_grid = _get_grouped_data_and_base_points(attribute_value)

        roc_convex_hull = _get_roc(grouped_data, x_grid, attribute_value)
        curve = _interpolate_curve(roc_convex_hull, 'x', 'y', 'operation', x_grid)

        _assert_interpolated_points_are_between_base_points(base_points, curve, ignore_for_base_points)

@pytest.mark.parametrize('roc_curve_based_post_processing_by_metric',
                        [roc_curve_based_post_processing_demographic_parity,
                         roc_curve_based_post_processing_equalized_odds])
def test_roc_curve_based_post_processing_non_binary_labels(
        roc_curve_based_post_processing_by_metric):
    non_binary_labels = copy.deepcopy(example_labels)
    non_binary_labels[0] = 2
    with pytest.raises(ValueError, match=NON_BINARY_LABELS_ERROR_MESSAGE):
        roc_curve_based_post_processing_by_metric(example_attributes1,
                                                  non_binary_labels,
                                                  example_scores)

@pytest.mark.parametrize('roc_curve_based_post_processing_by_metric',
                        [roc_curve_based_post_processing_demographic_parity,
                         roc_curve_based_post_processing_equalized_odds])
def test_roc_curve_based_post_processing_different_input_lengths(
        roc_curve_based_post_processing_by_metric):
    # try all combinations of input lists being shorter/longer than others
    n = len(example_attributes1)
    for permutation in _generate_list_reduction_permutations():
        with pytest.raises(ValueError, match=DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE):
            roc_curve_based_post_processing_by_metric(example_attributes1[:n-permutation[0]],
                                                      example_labels[:n-permutation[1]],
                                                      example_scores[:n-permutation[2]])

    # try providing empty lists in all combinations
    for permutation in _generate_empty_list_permutations():
        with pytest.raises(ValueError, match=EMPTY_INPUT_ERROR_MESSAGE):
            roc_curve_based_post_processing_by_metric(example_attributes1[:permutation[0]],
                                                      example_labels[:permutation[1]],
                                                      example_scores[:permutation[2]])

def test_roc_curve_based_post_processing_demographic_parity():
    adjusted_model = roc_curve_based_post_processing_demographic_parity(example_attributes1, example_labels, example_scores)

    # For Demographic Parity we can ignore p_ignore since it's always 0.

    # attribute value A
    value_for_less_than_2_5 = 0.8008
    assert np.isclose(value_for_less_than_2_5, adjusted_model(example_attribute_names1[0], 0))
    assert np.isclose(value_for_less_than_2_5, adjusted_model(example_attribute_names1[0], 2.499))
    assert 0 == adjusted_model(example_attribute_names1[0], 2.5)
    assert 0 == adjusted_model(example_attribute_names1[0], 100)

    # attribute value B
    value_for_less_than_0_5 = 0.00133333333333
    assert np.isclose(value_for_less_than_0_5, adjusted_model(example_attribute_names1[1], 0))
    assert np.isclose(value_for_less_than_0_5, adjusted_model(example_attribute_names1[1], 0.5))
    assert 1 == adjusted_model(example_attribute_names1[1], 0.51)
    assert 1 == adjusted_model(example_attribute_names1[1], 1)
    assert 1 == adjusted_model(example_attribute_names1[1], 100)

    # attribute value C
    value_between_0_5_and_1_5 = 0.608
    assert 0 == adjusted_model(example_attribute_names1[2], 0)
    assert 0 == adjusted_model(example_attribute_names1[2], 0.5)
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model(example_attribute_names1[2], 0.51))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model(example_attribute_names1[2], 1))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model(example_attribute_names1[2], 1.5))
    assert 1 == adjusted_model(example_attribute_names1[2], 1.51)
    assert 1 == adjusted_model(example_attribute_names1[2], 100)

    # Assert Demographic Parity actually holds
    discretized_predictions = _get_discretized_predictions(adjusted_model)

    # TODO check whether this is expected
    assert [sum([lp.prediction for lp in discretized_predictions[attribute_value]])
            / len(discretized_predictions[attribute_value])
            for attribute_value in discretized_predictions] == [5/7, 4/7, 5/6]

def test_roc_curve_based_post_processing_equalized_odds():
    adjusted_model = roc_curve_based_post_processing_equalized_odds(example_attributes1, example_labels, example_scores)

    # For Equalized Odds we need to factor in that the output is calculated by
    # p_ignore * prediction_constant + (1 - p_ignore) * (p0 * pred0(x) + p1 * pred1(x))
    # with p_ignore != 0 and prediction_constant != 0 for at least some attributes values.
    prediction_constant = 0.334

    # attribute value A
    # p_ignore is almost 0 which means there's almost no adjustment
    p_ignore = 0.001996007984031716
    base_value = prediction_constant * p_ignore
    value_for_less_than_2_5 = base_value + (1 - p_ignore) * 0.668
    assert np.isclose(value_for_less_than_2_5, adjusted_model(example_attribute_names1[0], 0))
    assert np.isclose(value_for_less_than_2_5, adjusted_model(example_attribute_names1[0], 2.499))
    assert base_value == adjusted_model(example_attribute_names1[0], 2.5)
    assert base_value == adjusted_model(example_attribute_names1[0], 100)

    # attribute value B
    # p_ignore is the largest among the three classes indicating a large adjustment
    p_ignore = 0.1991991991991991
    base_value = prediction_constant * p_ignore
    value_for_less_than_0_5 = base_value + (1 - p_ignore) * 0.001
    assert np.isclose(value_for_less_than_0_5, adjusted_model(example_attribute_names1[1], 0))
    assert np.isclose(value_for_less_than_0_5, adjusted_model(example_attribute_names1[1], 0.5))
    assert base_value + 1 - p_ignore == adjusted_model(example_attribute_names1[1], 0.51)
    assert base_value + 1 - p_ignore == adjusted_model(example_attribute_names1[1], 1)
    assert base_value + 1 - p_ignore == adjusted_model(example_attribute_names1[1], 100)

    # attribute value C
    # p_ignore is 0 which means there's no adjustment
    p_ignore = 0
    base_value = prediction_constant * p_ignore
    value_between_0_5_and_1_5 = base_value + (1 - p_ignore) * 0.501
    assert base_value == adjusted_model(example_attribute_names1[2], 0)
    assert base_value == adjusted_model(example_attribute_names1[2], 0.5)
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model(example_attribute_names1[2], 0.51))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model(example_attribute_names1[2], 1))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model(example_attribute_names1[2], 1.5))
    assert base_value + 1 - p_ignore == adjusted_model(example_attribute_names1[2], 1.51)
    assert base_value + 1 - p_ignore == adjusted_model(example_attribute_names1[2], 100)

    # Assert Equalized Odds actually holds
    discretized_predictions = _get_discretized_predictions(adjusted_model)

    # TODO check whether this is expected
    predictions_based_on_label = {}
    for label in [0, 1]:
        predictions_based_on_label[label] = \
            [sum([lp.prediction for lp in discretized_predictions[attribute_value] if lp.label == label])
             / len([lp for lp in discretized_predictions[attribute_value] if lp.label == label])
             for attribute_value in discretized_predictions]

    # assert counts of positive predictions for negative labels
    assert predictions_based_on_label[0] == [2/4, 1/3, 2/3]
    # assert counts of positive predictions for positive labels
    assert predictions_based_on_label[1] == [3/3, 3/4, 3/3]

def _assert_interpolated_points_are_between_base_points(base_points, curve, ignore_for_base_points=None):
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

    current_base_point_x, current_base_point_y = _get_base_point_coordinates(start_base_point_index, base_points)
    next_base_point_x, next_base_point_y = _get_base_point_coordinates(base_point_index, base_points)

    for x_grid_index in range(len(curve)):
        x = curve.x[x_grid_index]
        y = curve.y[x_grid_index]
        if np.isclose(x, current_base_point_x):
            assert np.isclose(y, current_base_point_y)
            continue

        while x > next_base_point_x:
            current_base_point_x, current_base_point_y = _get_base_point_coordinates(base_point_index, base_points)
            base_point_index += 1
            while base_point_index in ignore_for_base_points:
                base_point_index += 1
            next_base_point_x, next_base_point_y = _get_base_point_coordinates(base_point_index, base_points)

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
        assert np.isclose((y - current_base_point_y) / (x - current_base_point_x), (next_base_point_y - y) / (next_base_point_x - x))

def _assert_equal_points(expected_points, actual_points, ignore_indices=None):
    if ignore_indices is None:
        ignore_indices = []
    assert len(expected_points) - len(ignore_indices) == len(actual_points)

    # order by x to be able to iterate through
    actual_points = actual_points.sort_values(by="x")
    actual_points.index = range(len(actual_points))

    index_offset = 0
    for i in range(len(expected_points)):
        if i in ignore_indices:
            index_offset += 1

            if i > len(expected_points):
                break

            continue

        assert np.isclose(actual_points.x[i - index_offset], expected_points.x[i])
        assert np.isclose(actual_points.y[i - index_offset], expected_points.y[i])
        assert actual_points.operation[i - index_offset].operator == expected_points.operation[i].operator
        assert np.isclose(actual_points.operation[i - index_offset].threshold, expected_points.operation[i].threshold)

def _get_grouped_data_and_base_points(attribute_value):
    data = pd.DataFrame({ATTRIBUTE_KEY: example_attributes1, SCORE_KEY: example_scores, LABEL_KEY: example_labels})
    grouped_data = data.groupby(ATTRIBUTE_KEY).get_group(attribute_value).sort_values(by=SCORE_KEY, ascending=False)
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

def _generate_list_reduction_permutations():
    list_reduction_permutations = []
    for permutation in permutations([0, 0, 1]):
        list_reduction_permutations.append(permutation)
    for permutation in permutations([0, 1, 1]):
        list_reduction_permutations.append(permutation)

    return list_reduction_permutations

def _generate_empty_list_permutations():
    empty_list_permutations = []

    n = len(example_attributes1)
    for permutation in permutations([0, 0, n]):
        empty_list_permutations.append(permutation)
    for permutation in permutations([0, n, n]):
        empty_list_permutations.append(permutation)

    return empty_list_permutations

def _get_discretized_predictions(adjusted_model):
    labels_and_predictions = defaultdict(list)
    for i in range(len(example_attributes1)):
        labels_and_predictions[example_attributes1[i]].append(LabelAndPrediction(example_labels[i], adjusted_model(example_attributes1[i], example_scores[i])))

    return {
        attribute_value: [
            LabelAndPrediction(lp.label, int(lp.prediction >= 0.5)) for lp in labels_and_predictions[attribute_value]
        ] for attribute_value in labels_and_predictions
    }
