# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fairlearn.post_processing.roc_curve_based_post_processing import (roc_curve_based_post_processing,
                                                                       ThresholdOperation,
                                                                       _interpolate_curve,
                                                                       _get_roc,
                                                                       _calculate_roc_points,
                                                                       _get_scores_labels_and_counts,
                                                                       _filter_points_to_get_convex_hull,
                                                                       SCORE_KEY,
                                                                       LABEL_KEY,
                                                                       ATTRIBUTE_KEY)

example_attributes1 = [x for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
example_attributes2 = [x for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']
example_labels = [int(x) for x in '0110100' '0010111' '000111']
example_scores = [int(x) for x in '0011233' '0001111' '011112']

'''
def run_roc_curve_based_post_processing_and_plot(ex_attrs, flip):
    print("STARTING TEST")
    pred_EO = roc_curve_based_post_processing(ex_attrs, example_labels, example_scores, "EO", debug=True, flip=flip)
    pred_DP = roc_curve_based_post_processing(ex_attrs, example_labels, example_scores, "DP", debug=True, flip=flip)
    ex_preds_EO = []
    ex_preds_DP = []
    for i in range(len(ex_attrs)):
        ex_preds_EO.append( pred_EO(ex_attrs[i], example_scores[i]) )
        ex_preds_DP.append( pred_DP(ex_attrs[i], example_scores[i]) )
    ex_data = pd.DataFrame({ATTRIBUTE_KEY: ex_attrs, SCORE_KEY: example_scores, LABEL_KEY: example_labels, 'pred_EO': ex_preds_EO, 'pred_DP': ex_preds_DP})
    ex_data['error_EO'] = np.absolute(ex_data[LABEL_KEY]-ex_data['pred_EO'])
    ex_data['error_DP'] = np.absolute(ex_data[LABEL_KEY]-ex_data['pred_DP'])

    print("APPLYING EO PREDICTOR")
    print("")
    print(ex_data.groupby([ATTRIBUTE_KEY, LABEL_KEY]).mean()[['pred_EO']])
    print("")
    print("error_EO=%.3f" % ex_data['error_EO'].mean())

    print("-"*65)

    print("APPLYING DP PREDICTOR")
    print("")
    print(ex_data.groupby([ATTRIBUTE_KEY]).mean()[['pred_DP']])
    print("")
    print("error_DP=%.3f" % ex_data['error_DP'].mean())
    plt.show()

def test_1():
    run_roc_curve_based_post_processing_and_plot(example_attributes1, True)

def test_2():
    run_roc_curve_based_post_processing_and_plot(list(zip(example_attributes1, example_attributes2)), True)

def test_3():
    run_roc_curve_based_post_processing_and_plot(example_attributes1, False)

def test_4():
    run_roc_curve_based_post_processing_and_plot(example_attributes2, False)
'''

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

def _assert_interpolated_points_are_between_base_points(base_points, curve):
    def _get_base_point_coordinates(i, data):
        return data["x"][i], data["y"][i]

    print(base_points)
    print(curve)

    base_point_index = 1
    current_base_point_x, current_base_point_y = _get_base_point_coordinates(0, base_points)
    next_base_point_x, next_base_point_y = _get_base_point_coordinates(1, base_points)
    for x_grid_index in range(len(curve)):
        x = curve["x"][x_grid_index]
        y = curve["y"][x_grid_index]
        if x == current_base_point_x:
            assert y == current_base_point_y
            continue

        while x > next_base_point_x:
            current_base_point_x, current_base_point_y = _get_base_point_coordinates(base_point_index, base_points)
            base_point_index += 1
            next_base_point_x, next_base_point_y = _get_base_point_coordinates(base_point_index, base_points)
        
        if x == next_base_point_x:
            assert y == next_base_point_y
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

        assert np.isclose(actual_points["x"][i - index_offset], expected_points["x"][i])
        assert np.isclose(actual_points["y"][i - index_offset], expected_points["y"][i])
        assert actual_points["operation"][i - index_offset].operator == expected_points["operation"][i].operator
        assert np.isclose(actual_points["operation"][i - index_offset].threshold, expected_points["operation"][i].threshold)

def test_calculate_roc_points():
    data = pd.DataFrame({ATTRIBUTE_KEY: example_attributes1, SCORE_KEY: example_scores, LABEL_KEY: example_labels})
    grouped_data = data.groupby(ATTRIBUTE_KEY).get_group("A").sort_values(by=SCORE_KEY, ascending=False)

    roc_points = _calculate_roc_points(*_get_scores_labels_and_counts(grouped_data))
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
    _assert_equal_points(expected_roc_points, selected_points, ignore_indices=[1,2])

def test_get_roc():
    data = pd.DataFrame({ATTRIBUTE_KEY: example_attributes1, SCORE_KEY: example_scores, LABEL_KEY: example_labels})
    grouped_data = data.groupby(ATTRIBUTE_KEY).get_group("A").sort_values(by=SCORE_KEY, ascending=False)
    x_grid = np.linspace(0, 1, 100)
    
    roc, _ = _get_roc(grouped_data, x_grid, "A")

    base_points = pd.DataFrame({
        "x": [0, 0.5, 1],
        "y": [0, 1,   1],
        "operation": [ThresholdOperation('>', np.inf),
                      ThresholdOperation('<', 2.5),
                      ThresholdOperation('>', -np.inf)]
    })

    _assert_interpolated_points_are_between_base_points(base_points, roc)
