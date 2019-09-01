# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pytest
from fairlearn.post_processing.roc_curve_based_post_processing import \
    (roc_curve_based_post_processing_demographic_parity,
     roc_curve_based_post_processing_equalized_odds,
     DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE, EMPTY_INPUT_ERROR_MESSAGE,
     NON_BINARY_LABELS_ERROR_MESSAGE)
from .test_utilities import (example_attributes1, example_labels, example_scores,
                             example_attribute_names1, _generate_empty_list_permutations,
                             _get_discretized_predictions, _generate_list_reduction_permutations)


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
    adjusted_model = roc_curve_based_post_processing_demographic_parity(example_attributes1,
                                                                        example_labels,
                                                                        example_scores)

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

    assert [sum([lp.prediction for lp in discretized_predictions[attribute_value]])
            / len(discretized_predictions[attribute_value])
            for attribute_value in sorted(discretized_predictions)] == [5/7, 4/7, 5/6]


def test_roc_curve_based_post_processing_equalized_odds():
    adjusted_model = roc_curve_based_post_processing_equalized_odds(example_attributes1,
                                                                    example_labels,
                                                                    example_scores)

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

    predictions_based_on_label = {}
    for label in [0, 1]:
        predictions_based_on_label[label] = \
            [sum([lp.prediction for lp in discretized_predictions[attribute_value]
             if lp.label == label])
             / len([lp for lp in discretized_predictions[attribute_value] if lp.label == label])
             for attribute_value in sorted(discretized_predictions)]

    # assert counts of positive predictions for negative labels
    assert predictions_based_on_label[0] == [2/4, 1/3, 2/3]
    # assert counts of positive predictions for positive labels
    assert predictions_based_on_label[1] == [3/3, 3/4, 3/3]
