# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd
import pytest
from fairlearn.postprocessing._constants import DEMOGRAPHIC_PARITY, EQUALIZED_ODDS
from fairlearn._input_validation import \
    (_EMPTY_INPUT_ERROR_MESSAGE,
     _MESSAGE_X_NONE,
     _MESSAGE_Y_NONE,
     _MESSAGE_SENSITIVE_FEATURES_NONE,
     _MESSAGE_X_Y_ROWS,
     _MESSAGE_X_SENSITIVE_ROWS,
     _LABELS_NOT_0_1_ERROR_MESSAGE)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing._threshold_optimizer import \
    (_vectorized_prediction,
     _threshold_optimization_demographic_parity,
     _threshold_optimization_equalized_odds,
     NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE,
     PREDICT_BEFORE_FIT_ERROR_MESSAGE,
     MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE)
from fairlearn.postprocessing._postprocessing import \
    PREDICTOR_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE, EITHER_PREDICTOR_OR_ESTIMATOR_ERROR_MESSAGE, \
    MISSING_FIT_PREDICT_ERROR_MESSAGE, MISSING_PREDICT_ERROR_MESSAGE
from fairlearn.postprocessing._roc_curve_utilities import DEGENERATE_LABELS_ERROR_MESSAGE
from .conftest import (sensitive_features_ex1, sensitive_features_ex2,
                       sensitive_features_ex3, labels_ex, degenerate_labels_ex,
                       scores_ex, sensitive_feature_names_ex1, X_ex,
                       sensitive_feature_names_ex2, sensitive_feature_names_ex3,
                       _get_predictions_by_sensitive_feature, _map_into_single_column,
                       ExamplePredictor, ExampleEstimator, ExampleNotPredictor,
                       ExampleNotEstimator1, ExampleNotEstimator2,
                       candidate_A_transforms, candidate_X_transforms, candidate_Y_transforms,
                       is_invalid_sensitive_feature_transformation)


@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
@pytest.mark.parametrize("predict_method_name", ['predict', '_pmf_predict'])
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_predict_before_fit_error(X_transform, sensitive_features_transform, predict_method_name,
                                  constraints):
    X = X_transform(sensitive_features_ex1)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=constraints)

    with pytest.raises(ValueError, match=PREDICT_BEFORE_FIT_ERROR_MESSAGE):
        getattr(adjusted_predictor, predict_method_name)(X, sensitive_features=sensitive_features)


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_both_predictor_and_estimator_error(constraints):
    with pytest.raises(ValueError, match=EITHER_PREDICTOR_OR_ESTIMATOR_ERROR_MESSAGE):
        ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                           estimator=ExampleEstimator(),
                           constraints=constraints)


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_no_predictor_or_estimator_error(constraints):
    with pytest.raises(ValueError, match=PREDICTOR_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE):
        ThresholdOptimizer(constraints=constraints)


def test_constraints_not_supported():
    with pytest.raises(ValueError, match=NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE):
        ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                           constraints="UnsupportedConstraints")


@pytest.mark.parametrize("not_estimator", [ExampleNotEstimator1(), ExampleNotEstimator2()])
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_not_estimator(not_estimator, constraints):
    with pytest.raises(ValueError, match=MISSING_FIT_PREDICT_ERROR_MESSAGE):
        ThresholdOptimizer(estimator=not_estimator,
                           constraints=constraints)


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_not_predictor(constraints):
    with pytest.raises(ValueError, match=MISSING_PREDICT_ERROR_MESSAGE):
        ThresholdOptimizer(unconstrained_predictor=ExampleNotPredictor(),
                           constraints=constraints)


@pytest.mark.parametrize("X", [None, sensitive_features_ex1])
@pytest.mark.parametrize("y", [None, labels_ex])
@pytest.mark.parametrize("sensitive_features", [None, sensitive_features_ex1])
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_none_input_data(X, y, sensitive_features, constraints):
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=constraints)

    if y is None:
        with pytest.raises(ValueError) as exception:
            adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)
        assert str(exception.value) == _MESSAGE_Y_NONE
    elif X is None:
        with pytest.raises(ValueError) as exception:
            adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)
        assert "Expected 2D array, got scalar array instead" in str(exception.value)
    elif sensitive_features is None:
        with pytest.raises(ValueError) as exception:
            adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)
        assert str(exception.value) == _MESSAGE_SENSITIVE_FEATURES_NONE
    else:
        # skip since no arguments are None
        pass


@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_threshold_optimization_non_binary_labels(X_transform, y_transform,
                                                  sensitive_features_transform, constraints):
    non_binary_labels = copy.deepcopy(labels_ex)
    non_binary_labels[0] = 2

    X = X_transform(X_ex)
    y = y_transform(non_binary_labels)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)

    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=constraints)

    with pytest.raises(ValueError, match=_LABELS_NOT_0_1_ERROR_MESSAGE):
        adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)


@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_threshold_optimization_degenerate_labels(X_transform, y_transform,
                                                  sensitive_features_transform, constraints):
    X = X_transform(X_ex)
    y = y_transform(degenerate_labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)

    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=constraints)

    with pytest.raises(ValueError, match=DEGENERATE_LABELS_ERROR_MESSAGE.format('A')):
        adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)


@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
@pytest.mark.parametrize("sensitive_features_data",
                         [sensitive_features_ex1, sensitive_features_ex2, sensitive_features_ex3])
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_threshold_optimization_different_input_lengths(X_transform, y_transform,
                                                        sensitive_features_transform,
                                                        sensitive_features_data,
                                                        constraints):
    if is_invalid_sensitive_feature_transformation(sensitive_features_data, sensitive_features_transform):
        return

    n = len(X_ex)
    sensitive_features = sensitive_features_transform(sensitive_features_data)
    expected_exception_messages = {
        "inconsistent": 'Found input variables with inconsistent numbers of samples',
        "empty": 'Found array with 0 sample'
    }
    for permutation in [(0, 1), (1, 0)]:
        with pytest.raises(ValueError, match=expected_exception_messages['inconsistent']
                           .format("X, sensitive_features, and y")):
            X = X_transform(X_ex[:n - permutation[0]])
            y = y_transform(labels_ex[:n - permutation[1]])

            adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                                    constraints=constraints)
            adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)

    # try providing empty lists in all combinations
    for permutation in [(0, n, 'inconsistent'), (n, 0, 'empty')]:
        X = X_transform(X_ex[:n - permutation[0]])
        y = y_transform(labels_ex[:n - permutation[1]])

        adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                                constraints=constraints)
        with pytest.raises(ValueError, match=expected_exception_messages[permutation[2]]):
            adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)


@pytest.mark.parametrize("score_transform", candidate_Y_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
def test_threshold_optimization_demographic_parity(score_transform, y_transform,
                                                   sensitive_features_transform):
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    scores = score_transform(scores_ex)
    adjusted_predictor = create_adjusted_predictor(_threshold_optimization_demographic_parity,
                                                   sensitive_features, y, scores)

    # For Demographic Parity we can ignore p_ignore since it's always 0.

    # sensitive feature value A
    value_for_less_than_2_5 = 0.8008
    assert np.isclose(value_for_less_than_2_5,
                      adjusted_predictor([sensitive_feature_names_ex1[0]], [0]))
    assert np.isclose(value_for_less_than_2_5,
                      adjusted_predictor([sensitive_feature_names_ex1[0]], [2.499]))
    assert 0 == adjusted_predictor([sensitive_feature_names_ex1[0]], [2.5])
    assert 0 == adjusted_predictor([sensitive_feature_names_ex1[0]], [100])

    # sensitive feature value B
    value_for_less_than_0_5 = 0.00133333333333
    assert np.isclose(value_for_less_than_0_5,
                      adjusted_predictor([sensitive_feature_names_ex1[1]], [0]))
    assert np.isclose(value_for_less_than_0_5,
                      adjusted_predictor([sensitive_feature_names_ex1[1]], [0.5]))
    assert 1 == adjusted_predictor([sensitive_feature_names_ex1[1]], [0.51])
    assert 1 == adjusted_predictor([sensitive_feature_names_ex1[1]], [1])
    assert 1 == adjusted_predictor([sensitive_feature_names_ex1[1]], [100])

    # sensitive feature value C
    value_between_0_5_and_1_5 = 0.608
    assert 0 == adjusted_predictor([sensitive_feature_names_ex1[2]], [0])
    assert 0 == adjusted_predictor([sensitive_feature_names_ex1[2]], [0.5])
    assert np.isclose(value_between_0_5_and_1_5,
                      adjusted_predictor([sensitive_feature_names_ex1[2]], [0.51]))
    assert np.isclose(value_between_0_5_and_1_5,
                      adjusted_predictor([sensitive_feature_names_ex1[2]], [1]))
    assert np.isclose(value_between_0_5_and_1_5,
                      adjusted_predictor([sensitive_feature_names_ex1[2]], [1.5]))
    assert 1 == adjusted_predictor([sensitive_feature_names_ex1[2]], [1.51])
    assert 1 == adjusted_predictor([sensitive_feature_names_ex1[2]], [100])

    # Assert Demographic Parity actually holds
    predictions_by_sensitive_feature = _get_predictions_by_sensitive_feature(
        adjusted_predictor, sensitive_features_ex1, scores_ex, labels_ex)

    def _average_prediction(sensitive_feature_value, predictions_by_sensitive_feature):
        relevant_predictions = predictions_by_sensitive_feature[sensitive_feature_value]
        predictions = [lp.prediction for lp in relevant_predictions]
        return np.sum(predictions) / len(relevant_predictions)

    average_probabilities_by_sensitive_feature = []
    for sensitive_feature_value in sorted(predictions_by_sensitive_feature):
        average_probabilities_by_sensitive_feature \
            .append(_average_prediction(sensitive_feature_value,
                                        predictions_by_sensitive_feature))
    assert np.isclose(average_probabilities_by_sensitive_feature, [0.572] * 3).all()


@pytest.mark.parametrize("score_transform", candidate_Y_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
def test_threshold_optimization_equalized_odds(score_transform, y_transform,
                                               sensitive_features_transform):
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    scores = score_transform(scores_ex)
    adjusted_predictor = create_adjusted_predictor(_threshold_optimization_equalized_odds,
                                                   sensitive_features, y, scores)

    # For Equalized Odds we need to factor in that the output is calculated by
    # p_ignore * prediction_constant + (1 - p_ignore) * (p0 * pred0(x) + p1 * pred1(x))
    # with p_ignore != 0 and prediction_constant != 0 for at least some sensitive feature values.
    prediction_constant = 0.334

    # sensitive feature value A
    # p_ignore is almost 0 which means there's almost no adjustment
    p_ignore = 0.001996007984031716
    base_value = prediction_constant * p_ignore
    value_for_less_than_2_5 = base_value + (1 - p_ignore) * 0.668

    assert np.isclose(value_for_less_than_2_5,
                      adjusted_predictor([sensitive_feature_names_ex1[0]], [0]))
    assert np.isclose(value_for_less_than_2_5,
                      adjusted_predictor([sensitive_feature_names_ex1[0]], [2.499]))
    assert base_value == adjusted_predictor([sensitive_feature_names_ex1[0]], [2.5])
    assert base_value == adjusted_predictor([sensitive_feature_names_ex1[0]], [100])

    # sensitive feature value B
    # p_ignore is the largest among the three classes indicating a large adjustment
    p_ignore = 0.1991991991991991
    base_value = prediction_constant * p_ignore
    value_for_less_than_0_5 = base_value + (1 - p_ignore) * 0.001
    assert np.isclose(value_for_less_than_0_5,
                      adjusted_predictor([sensitive_feature_names_ex1[1]], [0]))
    assert np.isclose(value_for_less_than_0_5,
                      adjusted_predictor([sensitive_feature_names_ex1[1]], [0.5]))
    assert base_value + 1 - \
        p_ignore == adjusted_predictor([sensitive_feature_names_ex1[1]], [0.51])
    assert base_value + 1 - \
        p_ignore == adjusted_predictor([sensitive_feature_names_ex1[1]], [1])
    assert base_value + 1 - \
        p_ignore == adjusted_predictor([sensitive_feature_names_ex1[1]], [100])

    # sensitive feature value C
    # p_ignore is 0 which means there's no adjustment
    p_ignore = 0
    base_value = prediction_constant * p_ignore
    value_between_0_5_and_1_5 = base_value + (1 - p_ignore) * 0.501
    assert base_value == adjusted_predictor([sensitive_feature_names_ex1[2]], [0])
    assert base_value == adjusted_predictor([sensitive_feature_names_ex1[2]], [0.5])
    assert np.isclose(value_between_0_5_and_1_5,
                      adjusted_predictor([sensitive_feature_names_ex1[2]], [0.51]))
    assert np.isclose(value_between_0_5_and_1_5,
                      adjusted_predictor([sensitive_feature_names_ex1[2]], [1]))
    assert np.isclose(value_between_0_5_and_1_5,
                      adjusted_predictor([sensitive_feature_names_ex1[2]], [1.5]))
    assert base_value + 1 - \
        p_ignore == adjusted_predictor([sensitive_feature_names_ex1[2]], [1.51])
    assert base_value + 1 - \
        p_ignore == adjusted_predictor([sensitive_feature_names_ex1[2]], [100])

    # Assert Equalized Odds actually holds
    predictions_by_sensitive_feature = _get_predictions_by_sensitive_feature(
        adjusted_predictor, sensitive_features_ex1, scores_ex, labels_ex)

    def _average_prediction_for_label(label, sensitive_feature_value,
                                      predictions_by_sensitive_feature):
        relevant_predictions = predictions_by_sensitive_feature[sensitive_feature_value]
        predictions_for_label = [lp.prediction for lp in relevant_predictions if lp.label == label]
        sum_of_predictions_for_label = np.sum(predictions_for_label)
        n_predictions_for_label = len([lp for lp in relevant_predictions if lp.label == label])
        return sum_of_predictions_for_label / n_predictions_for_label

    predictions_based_on_label = {0: [], 1: []}
    for label in [0, 1]:
        for sensitive_feature_value in sorted(predictions_by_sensitive_feature):
            predictions_based_on_label[label] \
                .append(_average_prediction_for_label(label, sensitive_feature_value,
                        predictions_by_sensitive_feature))

    # assert counts of positive predictions for negative labels
    assert np.isclose(predictions_based_on_label[0], [0.334] * 3).all()
    # assert counts of positive predictions for positive labels
    assert np.isclose(predictions_based_on_label[1], [0.66733333] * 3).all()


@pytest.mark.parametrize("sensitive_feature_data,sensitive_feature_names,expected_p0,expected_p1",
                         [(sensitive_features_ex1, sensitive_feature_names_ex1, 0.428, 0.572),
                          (sensitive_features_ex2, sensitive_feature_names_ex2, 0.6, 0.4),
                          (sensitive_features_ex3, sensitive_feature_names_ex3, 0.5, 0.5)])
@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
def test_threshold_optimization_demographic_parity_e2e(
        sensitive_feature_data, sensitive_feature_names, expected_p0, expected_p1, X_transform,
        y_transform, sensitive_features_transform):

    if is_invalid_sensitive_feature_transformation(sensitive_feature_data, sensitive_features_transform):
        return

    X = X_transform(X_ex)
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_feature_data)

    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=DEMOGRAPHIC_PARITY, plot=False)
    adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)
    predictions = adjusted_predictor._pmf_predict(X, sensitive_features=sensitive_features)

    # assert demographic parity
    for sensitive_feature_name in sensitive_feature_names:
        average_probs = np.average(
            predictions[_map_into_single_column(sensitive_feature_data) == sensitive_feature_name], axis=0)
        assert np.isclose(average_probs[0], expected_p0)
        assert np.isclose(average_probs[1], expected_p1)


@pytest.mark.parametrize("sensitive_feature_data,sensitive_feature_names,"
                         "expected_positive_p0,expected_positive_p1,"
                         "expected_negative_p0,expected_negative_p1",
                         [(sensitive_features_ex1, sensitive_feature_names_ex1,
                           0.33266666, 0.66733333, 0.666, 0.334),
                          (sensitive_features_ex2, sensitive_feature_names_ex2,
                           0.112, 0.888, 0.334, 0.666)])
@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
def test_threshold_optimization_equalized_odds_e2e(
        sensitive_feature_data, sensitive_feature_names, expected_positive_p0,
        expected_positive_p1, expected_negative_p0, expected_negative_p1, X_transform,
        y_transform, sensitive_features_transform):

    if is_invalid_sensitive_feature_transformation(sensitive_feature_data, sensitive_features_transform):
        return

    X = X_transform(X_ex)
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_feature_data)
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=EQUALIZED_ODDS)
    adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)

    predictions = adjusted_predictor._pmf_predict(X, sensitive_features=sensitive_features)

    # assert equalized odds
    for a in sensitive_feature_names:
        positive_indices = (_map_into_single_column(sensitive_feature_data) == a) * (labels_ex == 1)
        negative_indices = (_map_into_single_column(sensitive_feature_data) == a) * (labels_ex == 0)
        average_probs_positive_indices = np.average(predictions[positive_indices], axis=0)
        average_probs_negative_indices = np.average(predictions[negative_indices], axis=0)
        assert np.isclose(average_probs_positive_indices[0], expected_positive_p0)
        assert np.isclose(average_probs_positive_indices[1], expected_positive_p1)
        assert np.isclose(average_probs_negative_indices[0], expected_negative_p0)
        assert np.isclose(average_probs_negative_indices[1], expected_negative_p1)


@pytest.mark.parametrize("sensitive_feature_data,sensitive_feature_names",
                         [(sensitive_features_ex1, sensitive_feature_names_ex1),
                          (sensitive_features_ex2, sensitive_feature_names_ex2)])
@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_predict_output_0_or_1(sensitive_feature_data, sensitive_feature_names, X_transform,
                               y_transform, sensitive_features_transform, constraints):
    X = X_transform(X_ex)
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_feature_data)
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=constraints)
    adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)

    predictions = adjusted_predictor.predict(X, sensitive_features=sensitive_features)
    for prediction in predictions:
        assert prediction in [0, 1]


@pytest.mark.parametrize("sensitive_feature_data,sensitive_feature_names",
                         [(sensitive_features_ex1, sensitive_feature_names_ex1),
                          (sensitive_features_ex2, sensitive_feature_names_ex2)])
@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)  # noqa: E501
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_predict_different_argument_lengths(sensitive_feature_data, sensitive_feature_names,
                                            X_transform, y_transform,
                                            sensitive_features_transform, constraints):
    X = X_transform(X_ex)
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_feature_data)
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(),
                                            constraints=constraints)
    adjusted_predictor.fit(X, y, sensitive_features=sensitive_features)

    with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples"):
        adjusted_predictor.predict(
            X, sensitive_features=sensitive_features_transform(sensitive_feature_data[:-1]))

    with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples"):
        adjusted_predictor.predict(X_transform(X_ex)[:-1],
                                   sensitive_features=sensitive_features)


def create_adjusted_predictor(threshold_optimization_method, sensitive_features, labels, scores):
    post_processed_predictor_by_sensitive_feature = threshold_optimization_method(
        sensitive_features, labels, scores)

    return lambda sensitive_features_, scores: _vectorized_prediction(
        post_processed_predictor_by_sensitive_feature, sensitive_features_, scores)
