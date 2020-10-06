# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.base import BaseEstimator

from copy import deepcopy
import numpy as np
import pytest
from fairlearn._input_validation import \
    (_MESSAGE_Y_NONE,
     _MESSAGE_SENSITIVE_FEATURES_NONE,
     _LABELS_NOT_0_1_ERROR_MESSAGE)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing._threshold_optimizer import \
    (NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE,
     BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
     )
from fairlearn.postprocessing._tradeoff_curve_utilities import DEGENERATE_LABELS_ERROR_MESSAGE
from .conftest import (sensitive_features_ex1, labels_ex, degenerate_labels_ex,
                       scores_ex, sensitive_feature_names_ex1, X_ex,
                       _get_predictions_by_sensitive_feature,
                       ExamplePredictor,
                       is_invalid_transformation,
                       candidate_A_transforms, candidate_X_transforms,
                       candidate_Y_transforms)
from test.unit.input_convertors import _map_into_single_column
import pandas as pd


@pytest.mark.parametrize("X_transform", candidate_X_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)
@pytest.mark.parametrize("predict_method_name", ['predict', '_pmf_predict'])
@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
def test_predict_before_fit_error(X_transform, sensitive_features_transform, predict_method_name,
                                  constraints):
    X = X_transform(sensitive_features_ex1)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    adjusted_predictor = ThresholdOptimizer(
        estimator=ExamplePredictor(scores_ex),
        constraints=constraints,
        prefit=False)

    with pytest.raises(ValueError, match='instance is not fitted yet'):
        getattr(adjusted_predictor, predict_method_name)(X, sensitive_features=sensitive_features)


@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
def test_no_estimator_error(constraints):
    with pytest.raises(ValueError, match=BASE_ESTIMATOR_NONE_ERROR_MESSAGE):
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
@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
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


@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
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
@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_threshold_optimization_degenerate_labels(data_X_sf, y_transform, constraints):
    y = y_transform(degenerate_labels_ex)

    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=constraints)

    feature_name = _degenerate_labels_feature_name[data_X_sf.example_name]
    with pytest.raises(ValueError, match=DEGENERATE_LABELS_ERROR_MESSAGE.format(feature_name)):
        adjusted_predictor.fit(data_X_sf.X, y,
                               sensitive_features=data_X_sf.sensitive_features)


@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
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


class PassThroughPredictor(BaseEstimator):
    def __init__(self, transform=None):
        self.transform = transform

    def fit(self, X, y=None, **kwargs):
        self.transform_ = self.transform
        return self

    def predict(self, X):
        if self.transform_ is None:
            return X[0]
        else:
            return self.transform_(X[0])


@pytest.mark.parametrize("score_transform", candidate_Y_transforms)
@pytest.mark.parametrize("y_transform", candidate_Y_transforms)
@pytest.mark.parametrize("sensitive_features_transform", candidate_A_transforms)
def test_threshold_optimization_demographic_parity(score_transform, y_transform,
                                                   sensitive_features_transform):
    y = y_transform(labels_ex)
    sensitive_features = sensitive_features_transform(sensitive_features_ex1)
    # PassThroughPredictor takes scores_ex as input in predict and
    # returns score_transform(scores_ex) as output
    estimator = ThresholdOptimizer(estimator=PassThroughPredictor(score_transform),
                                   constraints='demographic_parity',
                                   flip=True)
    estimator.fit(pd.DataFrame(scores_ex), y, sensitive_features=sensitive_features)

    def prob_pred(sensitive_features, scores):
        return estimator._pmf_predict(
            pd.DataFrame(scores), sensitive_features=sensitive_features)[0, 1]

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
    # PassThroughPredictor takes scores_ex as input in predict and
    # returns score_transform(scores_ex) as output
    estimator = ThresholdOptimizer(estimator=PassThroughPredictor(score_transform),
                                   constraints='equalized_odds',
                                   flip=True)
    estimator.fit(pd.DataFrame(scores_ex), y, sensitive_features=sensitive_features)

    def prob_pred(sensitive_features, scores):
        return estimator._pmf_predict(
            pd.DataFrame(scores), sensitive_features=sensitive_features)[0, 1]

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
                                            constraints='demographic_parity',
                                            flip=True)
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
                                            constraints='equalized_odds',
                                            flip=True)
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


@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
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


@pytest.mark.parametrize("constraints", ['demographic_parity', 'equalized_odds'])
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


constraints_list = [
    'selection_rate_parity',
    'demographic_parity',
    'false_positive_rate_parity',
    'false_negative_rate_parity',
    'true_positive_rate_parity',
    'true_negative_rate_parity',
    'equalized_odds',
    'bad_constraints']

objectives_list = [
    'accuracy_score',
    'balanced_accuracy_score',
    'selection_rate',
    'true_positive_rate',
    'true_negative_rate',
    'bad_objective']

# For each combination of constraints and objective,
# provide the returned solution as a dictionary, or
# a string that represents the ValueError if the combination
# is invalid.
results = {
    'selection_rate_parity, accuracy_score': {
        0: {'p0': 0.625, 'op0': '>', 'thr0': 1.5, 'p1': 0.375, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 0.5}},
    'selection_rate_parity, balanced_accuracy_score': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': 1.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.2, 'op0': '>', 'thr0': np.inf, 'p1': 0.8, 'op1': '>', 'thr1': 0.5}},
    'selection_rate_parity, selection_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'selection_rate_parity, true_positive_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': 1.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': 0.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'selection_rate_parity, true_negative_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 0.5}},
    'selection_rate_parity, bad_objective': (
        'For selection_rate_parity only the following objectives are supported'),
    'demographic_parity, accuracy_score': {
        0: {'p0': 0.625, 'op0': '>', 'thr0': 1.5, 'p1': 0.375, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 0.5}},
    'demographic_parity, balanced_accuracy_score': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': 1.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.2, 'op0': '>', 'thr0': np.inf, 'p1': 0.8, 'op1': '>', 'thr1': 0.5}},
    'demographic_parity, selection_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'demographic_parity, true_positive_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': 1.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': 0.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'demographic_parity, true_negative_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 0.5}},
    'demographic_parity, bad_objective': (
        'For demographic_parity only the following objectives are supported'),
    'false_positive_rate_parity, accuracy_score': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': 1.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 1.0, 'op0': '>', 'thr0': 0.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf}},
    'false_positive_rate_parity, balanced_accuracy_score': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': 1.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 1.0, 'op0': '>', 'thr0': 0.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf}},
    'false_positive_rate_parity, selection_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': 1.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': 0.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'false_positive_rate_parity, true_positive_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': 1.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': 0.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'false_positive_rate_parity, true_negative_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf}},
    'false_positive_rate_parity, bad_objective': (
        'For false_positive_rate_parity only the following objectives are supported'),
    'false_negative_rate_parity, accuracy_score': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 1.0, 'op0': '>', 'thr0': 0.5, 'p1': 0.0, 'op1': '>', 'thr1': np.inf}},
    'false_negative_rate_parity, balanced_accuracy_score': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': -np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 0.75, 'op0': '>', 'thr0': 0.5, 'p1': 0.25, 'op1': '>', 'thr1': np.inf}},
    'false_negative_rate_parity, selection_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 0.5},
        1: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': np.inf}},
    'false_negative_rate_parity, true_positive_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': np.inf},
        1: {'p0': 1.0, 'op0': '>', 'thr0': 0.5, 'p1': 0.0, 'op1': '>', 'thr1': np.inf}},
    'false_negative_rate_parity, true_negative_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': -np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 0.75, 'op0': '>', 'thr0': 0.5, 'p1': 0.25, 'op1': '>', 'thr1': np.inf}},
    'false_negative_rate_parity, bad_objective': (
        'For false_negative_rate_parity only the following objectives are supported'),
    'true_positive_rate_parity, accuracy_score': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': 1.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 0.5}},
    'true_positive_rate_parity, balanced_accuracy_score': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 0.25, 'op0': '>', 'thr0': np.inf, 'p1': 0.75, 'op1': '>', 'thr1': 0.5}},
    'true_positive_rate_parity, selection_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': 0.5, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'true_positive_rate_parity, true_positive_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p0': 0.0, 'op0': '>', 'thr0': np.inf, 'p1': 1.0, 'op1': '>', 'thr1': -np.inf}},
    'true_positive_rate_parity, true_negative_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 1.0, 'op0': '>', 'thr0': np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 0.5}},
    'true_positive_rate_parity, bad_objective': (
        'For true_positive_rate_parity only the following objectives are supported'),
    'true_negative_rate_parity, accuracy_score': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': -np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 0.0, 'op0': '>', 'thr0': -np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 0.5}},
    'true_negative_rate_parity, balanced_accuracy_score': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': -np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 0.0, 'op0': '>', 'thr0': -np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 0.5}},
    'true_negative_rate_parity, selection_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 0.5}},
    'true_negative_rate_parity, true_positive_rate': {
        0: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 1.0, 'op0': '>', 'thr0': -np.inf, 'p1': 0.0, 'op1': '>', 'thr1': 0.5}},
    'true_negative_rate_parity, true_negative_rate': {
        0: {'p0': 0.0, 'op0': '>', 'thr0': 0.5, 'p1': 1.0, 'op1': '>', 'thr1': 1.5},
        1: {'p0': 0.0, 'op0': '>', 'thr0': -np.inf, 'p1': 1.0, 'op1': '>', 'thr1': 0.5}},
    'true_negative_rate_parity, bad_objective': (
        'For true_negative_rate_parity only the following objectives are supported'),
    'equalized_odds, accuracy_score': {
        0: {'p_ignore': 0.0, 'prediction_constant': 0.0,
            'p0': 1.0, 'op0': '>', 'thr0': 1.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p_ignore': 0.25, 'prediction_constant': 0.0,
            'p0': 1.0, 'op0': '>', 'thr0': 0.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf}},
    'equalized_odds, balanced_accuracy_score': {
        0: {'p_ignore': 0.0, 'prediction_constant': 0.0,
            'p0': 1.0, 'op0': '>', 'thr0': 1.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf},
        1: {'p_ignore': 0.25, 'prediction_constant': 0.0,
            'p0': 1.0, 'op0': '>', 'thr0': 0.5, 'p1': 0.0, 'op1': '>', 'thr1': -np.inf}},
    'equalized_odds, selection_rate': (
        'For equalized_odds only the following objectives are supported'),
    'equalized_odds, true_positive_rate': (
        'For equalized_odds only the following objectives are supported'),
    'equalized_odds, true_negative_rate': (
        'For equalized_odds only the following objectives are supported'),
    'equalized_odds, bad_objective': (
        'For equalized_odds only the following objectives are supported'),
    'bad_constraints, accuracy_score': (
        'Currently only the following constraints are supported'),
    'bad_constraints, balanced_accuracy_score': (
        'Currently only the following constraints are supported'),
    'bad_constraints, selection_rate': (
        'Currently only the following constraints are supported'),
    'bad_constraints, true_positive_rate': (
        'Currently only the following constraints are supported'),
    'bad_constraints, true_negative_rate': (
        'Currently only the following constraints are supported'),
    'bad_constraints, bad_objective': (
        'Currently only the following constraints are supported')}

PREC = 1e-6


@pytest.mark.parametrize("constraints", constraints_list)
@pytest.mark.parametrize("objective", objectives_list)
def test_constraints_objective_pairs(constraints, objective):
    X = pd.Series(
        [0, 1, 2, 3, 4, 0, 1, 2, 3]).to_frame()
    sf = pd.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1])
    y = pd.Series(
        [1, 0, 1, 1, 1, 0, 1, 1, 1])
    thr_optimizer = ThresholdOptimizer(
        estimator=PassThroughPredictor(),
        constraints=constraints,
        objective=objective,
        grid_size=20)
    expected = results[constraints+", "+objective]
    if type(expected) is str:
        with pytest.raises(ValueError) as error_info:
            thr_optimizer.fit(X, y, sensitive_features=sf)
        assert str(error_info.value).startswith(expected)
    else:
        thr_optimizer.fit(X, y, sensitive_features=sf)
        res = thr_optimizer.interpolated_thresholder_.interpolation_dict
        for key in [0, 1]:
            assert res[key]['p0'] == pytest.approx(expected[key]['p0'], PREC)
            assert res[key]['operation0']._operator == expected[key]['op0']
            assert res[key]['operation0']._threshold == pytest.approx(expected[key]['thr0'], PREC)
            assert res[key]['p1'] == pytest.approx(expected[key]['p1'], PREC)
            assert res[key]['operation1']._operator == expected[key]['op1']
            assert res[key]['operation1']._threshold == pytest.approx(expected[key]['thr1'], PREC)
            if 'p_ignore' in expected[key]:
                assert res[key]['p_ignore'] == pytest.approx(expected[key]['p_ignore'], PREC)
                assert res[key]['prediction_constant'] == \
                    pytest.approx(expected[key]['prediction_constant'], PREC)
            else:
                assert 'p_ignore' not in res[key]
