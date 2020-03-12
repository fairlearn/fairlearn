# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from copy import deepcopy
import numpy as np
import pytest
from fairlearn.postprocessing._constants import DEMOGRAPHIC_PARITY, EQUALIZED_ODDS
from fairlearn._input_validation import \
    (_MESSAGE_Y_NONE,
     _MESSAGE_SENSITIVE_FEATURES_NONE,
     _LABELS_NOT_0_1_ERROR_MESSAGE)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing._threshold_optimizer import \
    (_vectorized_prediction,
     NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE,
     ESTIMATOR_ERROR_MESSAGE,
     )
from fairlearn.postprocessing._roc_curve_utilities import DEGENERATE_LABELS_ERROR_MESSAGE
from .conftest import (sensitive_features_ex1, labels_ex, degenerate_labels_ex,
                       scores_ex, sensitive_feature_names_ex1, X_ex,
                       _get_predictions_by_sensitive_feature,
                       ExamplePredictor,
                       is_invalid_transformation,
                       candidate_A_transforms, candidate_X_transforms,
                       candidate_Y_transforms)
from test.unit.input_convertors import _map_into_single_column


@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)
@pytest.mark.parametrize("predict_method_name", ['predict', '_pmf_predict'])
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_predict_before_fit_error(X_transform, sensitive_features_transform, predict_method_name,
                                  constraints):
    X = X_transform(sensitive_features_ex1)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    adjusted_predictor = ThresholdOptimizer(
        estimator=ExamplePredictor(scores_ex),
        constraints=constraints,
        prefit=True)

    with pytest.raises(ValueError, match='instance is not fitted yet'):
        getattr(adjusted_predictor, predict_method_name)(X, sensitive_features=sensitive_features)


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_no_estimator_error(constraints):
    with pytest.raises(ValueError, match=ESTIMATOR_ERROR_MESSAGE):
        ThresholdOptimizer(constraints=constraints).fit(
            X_ex, labels_ex, sensitive_features=sensitive_features_ex1)


def test_constraints_not_supported():
    with pytest.raises(ValueError, match=NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE):
        ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                           constraints="UnsupportedConstraints").fit(
                               X_ex, labels_ex,
                               sensitive_features=sensitive_features_ex1
                           )


@pytest.mark.parametrize("X", [None, X_ex])
@pytest.mark.parametrize("y", [None, labels_ex])
@pytest.mark.parametrize("sensitive_features", [None, sensitive_features_ex1])
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
def test_none_input_data(X, y, sensitive_features, constraints):
    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
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


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_threshold_optimization_non_binary_labels(data_X_y_sf, constraints):
    non_binary_y = deepcopy(data_X_y_sf.y)
    non_binary_y[0] = 2

    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=constraints)

    with pytest.raises(ValueError, match=_LABELS_NOT_0_1_ERROR_MESSAGE):
        adjusted_predictor.fit(data_X_y_sf.X, non_binary_y,
                               sensitive_features=data_X_y_sf.sensitive_features)


_degenerate_labels_feature_name = {
    "example 1": "A",
    "example 2": "Y",
    "example 3": "A,Y"
}


@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_threshold_optimization_degenerate_labels(data_X_sf, y_transform, constraints):
    y = y_transform(degenerate_labels_ex)

    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=constraints,
                                            prefit=True)

    feature_name = _degenerate_labels_feature_name[data_X_sf.example_name]
    with pytest.raises(ValueError, match=DEGENERATE_LABELS_ERROR_MESSAGE.format(feature_name)):
        adjusted_predictor.fit(data_X_sf.X, y,
                               sensitive_features=data_X_sf.sensitive_features)


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_threshold_optimization_different_input_lengths(data_X_y_sf, constraints):
    n = len(X_ex)
    expected_exception_messages = {
        "inconsistent": 'Found input variables with inconsistent numbers of samples',
        "empty": 'Found array with 0 sample'
    }
    for permutation in [(0, 1), (1, 0)]:
        with pytest.raises(ValueError, match=expected_exception_messages['inconsistent']
                           .format("X, sensitive_features, and y")):
            adjusted_predictor = ThresholdOptimizer(
                estimator=ExamplePredictor(scores_ex),
                constraints=constraints)
            adjusted_predictor.fit(data_X_y_sf.X[:n - permutation[0]],
                                   data_X_y_sf.y[:n - permutation[1]],
                                   sensitive_features=data_X_y_sf.sensitive_features)

    # try providing empty lists in all combinations
    for permutation in [(0, n, 'inconsistent'), (n, 0, 'empty')]:
        adjusted_predictor = ThresholdOptimizer(
            estimator=ExamplePredictor(scores_ex),
            constraints=constraints)
        with pytest.raises(ValueError, match=expected_exception_messages[permutation[2]]):
            adjusted_predictor.fit(data_X_y_sf.X[:n - permutation[0]],
                                   data_X_y_sf.y[:n - permutation[1]],
                                   sensitive_features=data_X_y_sf.sensitive_features)


@pytest.mark.parametrize("score_transform", candidate_Y_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)
def test_threshold_optimization_demographic_parity(score_transform, y_transform,
                                                   sensitive_features_transform):
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    scores = score_transform(scores_ex)
    estimator = ThresholdOptimizer(estimator=ExamplePredictor(scores),
                                   constraints=DEMOGRAPHIC_PARITY)
    estimator.fit(X_ex, y, sensitive_features=sensitive_features)

    def prob_pred(sensitive_features, scores):
        return _vectorized_prediction(
            estimator._post_processed_predictor_by_sensitive_feature,
            sensitive_features,
            scores)

    # For Demographic Parity we can ignore p_ignore since it's always 0.

    # sensitive feature value A
    value_for_less_than_2_5 = 0.8008
    assert np.isclose(value_for_less_than_2_5,
                      prob_pred([sensitive_feature_names_ex1[0]], [0]))
    assert np.isclose(value_for_less_than_2_5,
                      prob_pred([sensitive_feature_names_ex1[0]], [2.499]))
    assert 0 == prob_pred([sensitive_feature_names_ex1[0]], [2.5])
    assert 0 == prob_pred([sensitive_feature_names_ex1[0]], [100])

    # sensitive feature value B
    value_for_less_than_0_5 = 0.00133333333333
    assert np.isclose(value_for_less_than_0_5,
                      prob_pred([sensitive_feature_names_ex1[1]], [0]))
    assert np.isclose(value_for_less_than_0_5,
                      prob_pred([sensitive_feature_names_ex1[1]], [0.5]))
    assert 1 == prob_pred([sensitive_feature_names_ex1[1]], [0.51])
    assert 1 == prob_pred([sensitive_feature_names_ex1[1]], [1])
    assert 1 == prob_pred([sensitive_feature_names_ex1[1]], [100])

    # sensitive feature value C
    value_between_0_5_and_1_5 = 0.608
    assert 0 == prob_pred([sensitive_feature_names_ex1[2]], [0])
    assert 0 == prob_pred([sensitive_feature_names_ex1[2]], [0.5])
    assert np.isclose(value_between_0_5_and_1_5,
                      prob_pred([sensitive_feature_names_ex1[2]], [0.51]))
    assert np.isclose(value_between_0_5_and_1_5,
                      prob_pred([sensitive_feature_names_ex1[2]], [1]))
    assert np.isclose(value_between_0_5_and_1_5,
                      prob_pred([sensitive_feature_names_ex1[2]], [1.5]))
    assert 1 == prob_pred([sensitive_feature_names_ex1[2]], [1.51])
    assert 1 == prob_pred([sensitive_feature_names_ex1[2]], [100])

    # Assert Demographic Parity actually holds
    predictions_by_sensitive_feature = _get_predictions_by_sensitive_feature(
        prob_pred, sensitive_features_ex1, scores_ex, labels_ex)

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
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)
def test_threshold_optimization_equalized_odds(score_transform, y_transform,
                                               sensitive_features_transform):
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    scores = score_transform(scores_ex)
    estimator = ThresholdOptimizer(estimator=ExamplePredictor(scores),
                                   constraints=EQUALIZED_ODDS)
    estimator.fit(X_ex, y, sensitive_features=sensitive_features)

    def prob_pred(sensitive_features, scores):
        return _vectorized_prediction(
            estimator._post_processed_predictor_by_sensitive_feature,
            sensitive_features,
            scores)

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
                      prob_pred([sensitive_feature_names_ex1[0]], [0]))
    assert np.isclose(value_for_less_than_2_5,
                      prob_pred([sensitive_feature_names_ex1[0]], [2.499]))
    assert base_value == prob_pred([sensitive_feature_names_ex1[0]], [2.5])
    assert base_value == prob_pred([sensitive_feature_names_ex1[0]], [100])

    # sensitive feature value B
    # p_ignore is the largest among the three classes indicating a large adjustment
    p_ignore = 0.1991991991991991
    base_value = prediction_constant * p_ignore
    value_for_less_than_0_5 = base_value + (1 - p_ignore) * 0.001
    assert np.isclose(value_for_less_than_0_5,
                      prob_pred([sensitive_feature_names_ex1[1]], [0]))
    assert np.isclose(value_for_less_than_0_5,
                      prob_pred([sensitive_feature_names_ex1[1]], [0.5]))
    assert base_value + 1 - \
        p_ignore == prob_pred([sensitive_feature_names_ex1[1]], [0.51])
    assert base_value + 1 - \
        p_ignore == prob_pred([sensitive_feature_names_ex1[1]], [1])
    assert base_value + 1 - \
        p_ignore == prob_pred([sensitive_feature_names_ex1[1]], [100])

    # sensitive feature value C
    # p_ignore is 0 which means there's no adjustment
    p_ignore = 0
    base_value = prediction_constant * p_ignore
    value_between_0_5_and_1_5 = base_value + (1 - p_ignore) * 0.501
    assert base_value == prob_pred([sensitive_feature_names_ex1[2]], [0])
    assert base_value == prob_pred([sensitive_feature_names_ex1[2]], [0.5])
    assert np.isclose(value_between_0_5_and_1_5,
                      prob_pred([sensitive_feature_names_ex1[2]], [0.51]))
    assert np.isclose(value_between_0_5_and_1_5,
                      prob_pred([sensitive_feature_names_ex1[2]], [1]))
    assert np.isclose(value_between_0_5_and_1_5,
                      prob_pred([sensitive_feature_names_ex1[2]], [1.5]))
    assert base_value + 1 - \
        p_ignore == prob_pred([sensitive_feature_names_ex1[2]], [1.51])
    assert base_value + 1 - \
        p_ignore == prob_pred([sensitive_feature_names_ex1[2]], [100])

    # Assert Equalized Odds actually holds
    predictions_by_sensitive_feature = _get_predictions_by_sensitive_feature(
        prob_pred, sensitive_features_ex1, scores_ex, labels_ex)

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


_P0 = "p0"
_P1 = "p1"
_expected_ps_demographic_parity = {
    "example 1": {
        _P0: 0.428,
        _P1: 0.572
    },
    "example 2": {
        _P0: 0.6,
        _P1: 0.4
    },
    "example 3": {
        _P0: 0.5,
        _P1: 0.5
    }
}


@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_threshold_optimization_demographic_parity_e2e(data_X_y_sf):
    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=DEMOGRAPHIC_PARITY)
    adjusted_predictor.fit(data_X_y_sf.X, data_X_y_sf.y,
                           sensitive_features=data_X_y_sf.sensitive_features)
    predictions = adjusted_predictor._pmf_predict(
        data_X_y_sf.X, sensitive_features=data_X_y_sf.sensitive_features)

    expected_ps = _expected_ps_demographic_parity[data_X_y_sf.example_name]

    # assert demographic parity
    for sensitive_feature_name in data_X_y_sf.feature_names:
        average_probs = np.average(
            predictions[
                _map_into_single_column(data_X_y_sf.sensitive_features) == sensitive_feature_name
            ], axis=0)
        assert np.isclose(average_probs[0], expected_ps[_P0])
        assert np.isclose(average_probs[1], expected_ps[_P1])


_POS_P0 = "positive_p0"
_POS_P1 = "positive_p1"
_NEG_P0 = "negative_p0"
_NEG_P1 = "negative_p1"
_expected_ps_equalized_odds = {
    "example 1": {
        _POS_P0: 0.33266666,
        _POS_P1: 0.66733333,
        _NEG_P0: 0.666,
        _NEG_P1: 0.334
    },
    "example 2": {
        _POS_P0: 0.112,
        _POS_P1: 0.888,
        _NEG_P0: 0.334,
        _NEG_P1: 0.666
    },
    "example 3": {
        _POS_P0: 0.33333333333333337,
        _POS_P1: 0.6666666666666666,
        _NEG_P0: 0.5,
        _NEG_P1: 0.5
    }
}


@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_threshold_optimization_equalized_odds_e2e(data_X_y_sf):
    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=EQUALIZED_ODDS)
    adjusted_predictor.fit(data_X_y_sf.X, data_X_y_sf.y,
                           sensitive_features=data_X_y_sf.sensitive_features)

    predictions = adjusted_predictor._pmf_predict(
        data_X_y_sf.X, sensitive_features=data_X_y_sf.sensitive_features)

    expected_ps = _expected_ps_equalized_odds[data_X_y_sf.example_name]
    mapped_sensitive_features = _map_into_single_column(data_X_y_sf.sensitive_features)

    # assert equalized odds
    for a in data_X_y_sf.feature_names:
        pos_indices = (mapped_sensitive_features == a) * (labels_ex == 1)
        neg_indices = (mapped_sensitive_features == a) * (labels_ex == 0)
        average_probs_positive_indices = np.average(predictions[pos_indices], axis=0)
        average_probs_negative_indices = np.average(predictions[neg_indices], axis=0)
        assert np.isclose(average_probs_positive_indices[0], expected_ps[_POS_P0])
        assert np.isclose(average_probs_positive_indices[1], expected_ps[_POS_P1])
        assert np.isclose(average_probs_negative_indices[0], expected_ps[_NEG_P0])
        assert np.isclose(average_probs_negative_indices[1], expected_ps[_NEG_P1])


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_predict_output_0_or_1(data_X_y_sf, constraints):
    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=constraints)
    adjusted_predictor.fit(data_X_y_sf.X, data_X_y_sf.y,
                           sensitive_features=data_X_y_sf.sensitive_features)

    predictions = adjusted_predictor.predict(
        data_X_y_sf.X, sensitive_features=data_X_y_sf.sensitive_features)
    for prediction in predictions:
        assert prediction in [0, 1]


@pytest.mark.parametrize("constraints", [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS])
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_predict_different_argument_lengths(data_X_y_sf, constraints):
    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=constraints)
    adjusted_predictor.fit(data_X_y_sf.X, data_X_y_sf.y,
                           sensitive_features=data_X_y_sf.sensitive_features)

    with pytest.raises(ValueError,
                       match="Found input variables with inconsistent numbers of samples"):
        adjusted_predictor.predict(data_X_y_sf.X,
                                   sensitive_features=data_X_y_sf.sensitive_features[:-1])

    with pytest.raises(ValueError,
                       match="Found input variables with inconsistent numbers of samples"):
        adjusted_predictor.predict(data_X_y_sf.X[:-1],
                                   sensitive_features=data_X_y_sf.sensitive_features)


def create_adjusted_predictor(threshold_optimization_method, sensitive_features, labels, scores):
    post_processed_predictor_by_sensitive_feature = threshold_optimization_method(
        sensitive_features, labels, scores)

    return lambda sensitive_features_, scores: _vectorized_prediction(
        post_processed_predictor_by_sensitive_feature, sensitive_features_, scores)
