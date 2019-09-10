# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import DemographicParity, EqualizedOdds
from fairlearn.post_processing.roc_curve_based_post_processing import \
    (ROCCurveBasedPostProcessing,
     _vectorized_prediction,
     _roc_curve_based_post_processing_demographic_parity,
     _roc_curve_based_post_processing_equalized_odds,
     DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE,
     EMPTY_INPUT_ERROR_MESSAGE,
     NON_BINARY_LABELS_ERROR_MESSAGE,
     INPUT_DATA_CONSISTENCY_ERROR_MESSAGE,
     MISSING_FIT_PREDICT_ERROR_MESSAGE,
     MISSING_PREDICT_ERROR_MESSAGE,
     FAIRNESS_METRIC_EXPECTED_ERROR_MESSAGE,
     NOT_SUPPORTED_FAIRNESS_METRIC_ERROR_MESSAGE,
     MODEL_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE,
     EITHER_MODEL_OR_ESTIMATOR_ERROR_MESSAGE,
     PREDICT_BEFORE_FIT_ERROR_MESSAGE)
from .test_utilities import (example_attributes1, example_attributes2, example_labels,
                             example_scores, example_attribute_names1, example_attribute_names2,
                             _generate_empty_list_permutations, _get_predictions_by_attribute,
                             _generate_list_reduction_permutations, _format_as_list_of_lists,
                             ExampleModel, ExampleEstimator, ExampleMetric, ExampleNotMetric,
                             ExampleNotModel, ExampleNotEstimator1, ExampleNotEstimator2)


@pytest.mark.parametrize("predict_method_name", ['predict', 'predict_proba'])
def test_predict_before_fit_error(predict_method_name):
    X, Y, A = _format_as_list_of_lists(example_attributes1), example_labels, example_attributes1
    adjusted_model = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                                 fairness_metric=DemographicParity())

    with pytest.raises(ValueError, match=PREDICT_BEFORE_FIT_ERROR_MESSAGE):
        _ = getattr(adjusted_model, predict_method_name)(X, A)


def test_both_model_and_estimator_error():
    with pytest.raises(ValueError, match=EITHER_MODEL_OR_ESTIMATOR_ERROR_MESSAGE):
        _ = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                        fairness_unaware_estimator=ExampleEstimator(),
                                        fairness_metric=DemographicParity())


def test_no_model_or_estimator_error():
    with pytest.raises(ValueError, match=MODEL_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE):
        _ = ROCCurveBasedPostProcessing(fairness_metric=DemographicParity())


def test_metric_not_supported():
    with pytest.raises(ValueError, match=NOT_SUPPORTED_FAIRNESS_METRIC_ERROR_MESSAGE):
        _ = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                        fairness_metric=ExampleMetric())


def test_not_a_fairness_metric():
    with pytest.raises(TypeError, match=FAIRNESS_METRIC_EXPECTED_ERROR_MESSAGE):
        _ = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                        fairness_metric=ExampleNotMetric())


@pytest.mark.parametrize("not_estimator", [ExampleNotEstimator1(), ExampleNotEstimator2()])
def test_not_estimator(not_estimator):
    with pytest.raises(ValueError, match=MISSING_FIT_PREDICT_ERROR_MESSAGE):
        _ = ROCCurveBasedPostProcessing(fairness_unaware_estimator=not_estimator,
                                        fairness_metric=ExampleMetric())


def test_not_model():
    with pytest.raises(ValueError, match=MISSING_PREDICT_ERROR_MESSAGE):
        _ = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleNotModel(),
                                        fairness_metric=ExampleMetric())


@pytest.mark.parametrize("X_transform,Y_transform,A_transform",
                         [(np.array, lambda Y: Y, lambda A: A),
                          (lambda X: X, np.array, lambda A: A),
                          (lambda X: X, lambda Y: Y, np.array),
                          (np.array, np.array, lambda A: A),
                          (lambda X: X, np.array, np.array),
                          (np.array, lambda Y: Y, np.array),
                          (pd.DataFrame, lambda Y: Y, lambda A: A),
                          (lambda X: X, pd.DataFrame, lambda A: A),
                          (lambda X: X, lambda Y: Y, pd.DataFrame),
                          (pd.DataFrame, pd.DataFrame, lambda A: A),
                          (lambda X: X, pd.DataFrame, pd.DataFrame),
                          (pd.DataFrame, lambda Y: Y, pd.DataFrame),
                          (pd.DataFrame, pd.DataFrame, pd.DataFrame)])
def test_inconsistent_input_data_types(X_transform, Y_transform, A_transform):
    X = X_transform(_format_as_list_of_lists(example_attributes1))
    Y = Y_transform(example_labels)
    A = A_transform(example_attributes1)
    adjusted_model = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                                 fairness_metric=DemographicParity())
    
    error_message = INPUT_DATA_CONSISTENCY_ERROR_MESSAGE.format(type(X).__name__,
                                                                type(Y).__name__,
                                                                type(A).__name__)

    with pytest.raises(ValueError) as exception:
        adjusted_model.fit(X, Y, A)
    assert str(exception.value) == error_message


@pytest.mark.parametrize("formatting_function", [lambda x: x, np.array])
@pytest.mark.parametrize('Metric', [DemographicParity, EqualizedOdds])
def test_roc_curve_based_post_processing_non_binary_labels(formatting_function, Metric):
    non_binary_labels = copy.deepcopy(example_labels)
    non_binary_labels[0] = 2

    X, Y, A = _format_X_Y_A(formatting_function, _format_as_list_of_lists(example_attributes1), non_binary_labels,
                            example_attributes1)
    adjusted_model = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                                 fairness_metric=Metric())
    
    with pytest.raises(ValueError, match=NON_BINARY_LABELS_ERROR_MESSAGE):
        adjusted_model.fit(X, Y, A)


@pytest.mark.parametrize("formatting_function", [lambda x: x, np.array])
@pytest.mark.parametrize('Metric', [DemographicParity, EqualizedOdds])
def test_roc_curve_based_post_processing_different_input_lengths(formatting_function, Metric):
    n = len(example_attributes1)
    for permutation in [(0, 1), (1, 0)]:
        with pytest.raises(ValueError, match=DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE):
            X, Y, A = _format_X_Y_A(formatting_function, _format_as_list_of_lists(example_attributes1)[:n-permutation[0]],
                                    example_labels[:n-permutation[1]],
                                    example_attributes1)
            adjusted_model = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                                         fairness_metric=Metric())
            adjusted_model.fit(X, Y, A)

    # try providing empty lists in all combinations
    for permutation in [(0, n), (n, 0)]:
        X, Y, A = _format_X_Y_A(formatting_function, _format_as_list_of_lists(example_attributes1)[:permutation[0]],
                                example_labels[:permutation[1]],
                                example_attributes1)

        adjusted_model = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                                     fairness_metric=Metric())
        with pytest.raises(ValueError, match=EMPTY_INPUT_ERROR_MESSAGE):
            adjusted_model.fit(X, Y, A)


@pytest.mark.parametrize('roc_curve_based_post_processing_by_metric',
                         [_roc_curve_based_post_processing_demographic_parity,
                          _roc_curve_based_post_processing_equalized_odds])
def test_roc_curve_based_post_processing_different_input_lengths_internals(
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
    adjusted_model = create_adjusted_model(_roc_curve_based_post_processing_demographic_parity,
                                           example_attributes1, example_labels, example_scores)

    # For Demographic Parity we can ignore p_ignore since it's always 0.

    # attribute value A
    value_for_less_than_2_5 = 0.8008
    assert np.isclose(value_for_less_than_2_5, adjusted_model([example_attribute_names1[0]], [0]))
    assert np.isclose(value_for_less_than_2_5, adjusted_model([example_attribute_names1[0]], [2.499]))
    assert 0 == adjusted_model([example_attribute_names1[0]], [2.5])
    assert 0 == adjusted_model([example_attribute_names1[0]], [100])

    # attribute value B
    value_for_less_than_0_5 = 0.00133333333333
    assert np.isclose(value_for_less_than_0_5, adjusted_model([example_attribute_names1[1]], [0]))
    assert np.isclose(value_for_less_than_0_5, adjusted_model([example_attribute_names1[1]], [0.5]))
    assert 1 == adjusted_model([example_attribute_names1[1]], [0.51])
    assert 1 == adjusted_model([example_attribute_names1[1]], [1])
    assert 1 == adjusted_model([example_attribute_names1[1]], [100])

    # attribute value C
    value_between_0_5_and_1_5 = 0.608
    assert 0 == adjusted_model([example_attribute_names1[2]], [0])
    assert 0 == adjusted_model([example_attribute_names1[2]], [0.5])
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model([example_attribute_names1[2]], [0.51]))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model([example_attribute_names1[2]], [1]))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model([example_attribute_names1[2]], [1.5]))
    assert 1 == adjusted_model([example_attribute_names1[2]], [1.51])
    assert 1 == adjusted_model([example_attribute_names1[2]], [100])

    # Assert Demographic Parity actually holds
    predictions_by_attribute = _get_predictions_by_attribute(adjusted_model, example_attributes1,
                                                             example_scores, example_labels)

    average_probabilities_by_attribute = \
        [np.sum([lp.prediction for lp in predictions_by_attribute[attribute_value]])
         / len(predictions_by_attribute[attribute_value])
         for attribute_value in sorted(predictions_by_attribute)]
    assert np.isclose(average_probabilities_by_attribute, [0.572] * 3).all()


def test_roc_curve_based_post_processing_equalized_odds():
    adjusted_model = create_adjusted_model(_roc_curve_based_post_processing_equalized_odds,
                                           example_attributes1, example_labels, example_scores)

    # For Equalized Odds we need to factor in that the output is calculated by
    # p_ignore * prediction_constant + (1 - p_ignore) * (p0 * pred0(x) + p1 * pred1(x))
    # with p_ignore != 0 and prediction_constant != 0 for at least some attributes values.
    prediction_constant = 0.334

    # attribute value A
    # p_ignore is almost 0 which means there's almost no adjustment
    p_ignore = 0.001996007984031716
    base_value = prediction_constant * p_ignore
    value_for_less_than_2_5 = base_value + (1 - p_ignore) * 0.668
    assert np.isclose(value_for_less_than_2_5, adjusted_model([example_attribute_names1[0]], [0]))
    assert np.isclose(value_for_less_than_2_5, adjusted_model([example_attribute_names1[0]], [2.499]))
    assert base_value == adjusted_model([example_attribute_names1[0]], [2.5])
    assert base_value == adjusted_model([example_attribute_names1[0]], [100])

    # attribute value B
    # p_ignore is the largest among the three classes indicating a large adjustment
    p_ignore = 0.1991991991991991
    base_value = prediction_constant * p_ignore
    value_for_less_than_0_5 = base_value + (1 - p_ignore) * 0.001
    assert np.isclose(value_for_less_than_0_5, adjusted_model([example_attribute_names1[1]], [0]))
    assert np.isclose(value_for_less_than_0_5, adjusted_model([example_attribute_names1[1]], [0.5]))
    assert base_value + 1 - p_ignore == adjusted_model([example_attribute_names1[1]], [0.51])
    assert base_value + 1 - p_ignore == adjusted_model([example_attribute_names1[1]], [1])
    assert base_value + 1 - p_ignore == adjusted_model([example_attribute_names1[1]], [100])

    # attribute value C
    # p_ignore is 0 which means there's no adjustment
    p_ignore = 0
    base_value = prediction_constant * p_ignore
    value_between_0_5_and_1_5 = base_value + (1 - p_ignore) * 0.501
    assert base_value == adjusted_model([example_attribute_names1[2]], [0])
    assert base_value == adjusted_model([example_attribute_names1[2]], [0.5])
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model([example_attribute_names1[2]], [0.51]))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model([example_attribute_names1[2]], [1]))
    assert np.isclose(value_between_0_5_and_1_5, adjusted_model([example_attribute_names1[2]], [1.5]))
    assert base_value + 1 - p_ignore == adjusted_model([example_attribute_names1[2]], [1.51])
    assert base_value + 1 - p_ignore == adjusted_model([example_attribute_names1[2]], [100])

    # Assert Equalized Odds actually holds
    predictions_by_attribute = _get_predictions_by_attribute(adjusted_model, example_attributes1,
                                                             example_scores, example_labels)

    predictions_based_on_label = {}
    for label in [0, 1]:
        predictions_based_on_label[label] = \
            [np.sum([lp.prediction for lp in predictions_by_attribute[attribute_value]
             if lp.label == label])
             / len([lp for lp in predictions_by_attribute[attribute_value] if lp.label == label])
             for attribute_value in sorted(predictions_by_attribute)]

    # assert counts of positive predictions for negative labels
    assert np.isclose(predictions_based_on_label[0], [0.334] * 3).all()
    # assert counts of positive predictions for positive labels
    assert np.isclose(predictions_based_on_label[1], [0.66733333] * 3).all()


@pytest.mark.parametrize("attributes,attribute_names,expected_p0,expected_p1",
                         [(example_attributes1, example_attribute_names1, 0.428, 0.572),
                          (example_attributes2, example_attribute_names2, 0.6, 0.4)])
@pytest.mark.parametrize("formatting_function", [lambda x: x, np.array])
def test_roc_curve_based_post_processing_demographic_parity_e2e(attributes, attribute_names,
                                                                expected_p0, expected_p1,
                                                                formatting_function):
    X, Y, A = _format_X_Y_A(formatting_function, _format_as_list_of_lists(attributes),
                            example_labels, attributes)
    adjusted_model = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                                 fairness_metric=DemographicParity())
    adjusted_model.fit(X, Y, A)

    predictions = adjusted_model.predict_proba(X, A)

    # assert demographic parity
    for a in attribute_names:
        average_probs = np.average(predictions[np.array(attributes) == a], axis=0)
        assert np.isclose(average_probs[0], expected_p0)
        assert np.isclose(average_probs[1], expected_p1)


@pytest.mark.parametrize("attributes,attribute_names,"
                         "expected_positive_p0,expected_positive_p1,"
                         "expected_negative_p0,expected_negative_p1",
                         [(example_attributes1, example_attribute_names1,
                           0.33266666, 0.66733333, 0.666, 0.334),
                          (example_attributes2, example_attribute_names2,
                           0.112, 0.888, 0.334, 0.666)])
@pytest.mark.parametrize("formatting_function", [lambda x: x, np.array])
def test_roc_curve_based_post_processing_equalized_odds_e2e(
        attributes, attribute_names, expected_positive_p0, expected_positive_p1,
        expected_negative_p0, expected_negative_p1, formatting_function):
    X, Y, A = _format_X_Y_A(formatting_function, _format_as_list_of_lists(attributes),
                            example_labels, attributes)
    adjusted_model = ROCCurveBasedPostProcessing(fairness_unaware_model=ExampleModel(),
                                                 fairness_metric=EqualizedOdds())
    adjusted_model.fit(X, Y, A)

    predictions = adjusted_model.predict_proba(X, A)

    # assert equalized odds
    for a in attribute_names:
        positive_indices = (np.array(attributes) == a) * (np.array(example_labels) == 1)
        negative_indices = (np.array(attributes) == a) * (np.array(example_labels) == 0)
        average_probs_positive_indices = np.average(predictions[positive_indices], axis=0)
        average_probs_negative_indices = np.average(predictions[negative_indices], axis=0)
        assert np.isclose(average_probs_positive_indices[0], expected_positive_p0)
        assert np.isclose(average_probs_positive_indices[1], expected_positive_p1)
        assert np.isclose(average_probs_negative_indices[0], expected_negative_p0)
        assert np.isclose(average_probs_negative_indices[1], expected_negative_p1)


def _format_X_Y_A(formatting_function, unformatted_X, unformatted_Y, unformatted_A):
    return formatting_function(unformatted_X), formatting_function(unformatted_Y), \
        formatting_function(unformatted_A)


def create_adjusted_model(roc_curve_based_post_processing_method, example_attributes,
                          example_labels, example_scores):
    post_processed_model_by_attribute = roc_curve_based_post_processing_method(
        example_attributes, example_labels, example_scores)

    return lambda A, scores: _vectorized_prediction(post_processed_model_by_attribute, A, scores)
