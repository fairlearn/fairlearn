# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" ROC Curve based Post processing algorithm based on M. Hardt,
E. Price, N. Srebro's paper "Equality of Opportunity in Supervised
Learning" (https://arxiv.org/pdf/1610.02413.pdf) for binary
classification with one categorical protected attribute.
"""

import logging
import numpy as np
import pandas as pd

from fairlearn.exceptions import NotFittedException
from fairlearn.metrics import FairnessMetric, DemographicParity, EqualizedOdds
from fairlearn.post_processing import PostProcessing
from ._constants import LABEL_KEY, SCORE_KEY, ATTRIBUTE_KEY
from ._roc_curve_utilities import _interpolate_curve, _get_roc
from ._roc_curve_plotting_utilities import plot_solution_and_show_plot, plot_overlap, plot_curve

OUTPUT_SEPARATOR = "-"*65

DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "Attributes, labels, and scores need to be of equal length."
EMPTY_INPUT_ERROR_MESSAGE = "At least one of attributes, labels, or scores are empty."
NON_BINARY_LABELS_ERROR_MESSAGE = "Labels other than 0/1 were provided."

logger = logging.getLogger(__name__)


class ROCCurveBasedPostProcessing(PostProcessing):
    def __init__(self, *, fairness_unaware_model=None, fairness_unaware_estimator=None,
                 fairness_metric=DemographicParity(), gridsize=1000, flip=True, plot=False):
        """
        Creates the post processing object.
        :param fairness_unaware_model: the trained model whose output will be post processed
        :type fairness_unaware_model: a trained model
        :param fairness_unaware_estimator: an untrained estimator that will be trained, and
            subsequently its output will be post processed
        :type fairness_unaware_estimator: an untrained estimator
        :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
        :type gridsize: int
        :param flip: allow flipping to negative weights if it improves accuracy.
        :type flip: bool
        :param plot: show ROC plot if True
        :type plot: bool
        """
        if fairness_unaware_model and fairness_unaware_estimator:
            raise ValueError("Only one of 'fairness_unaware_model' and "
                             "'fairness_unaware_estimator' can be passed.")
        elif fairness_unaware_model:
            self._fairness_unaware_model = fairness_unaware_model
            self._fairness_unaware_estimator = None
            self._validate_model()
        elif fairness_unaware_estimator:
            self._fairness_unaware_model = None
            self._fairness_unaware_estimator = fairness_unaware_estimator
            self._validate_estimator()
        else:
            raise ValueError("One of fairness_unaware_model and fairness_unaware_estimator' need "
                             "to be passed.")
        
        self._fairness_metric = fairness_metric
        self._validate_fairness_metric()
        
        self._gridsize = gridsize
        self._flip = flip
        self._plot = plot
        self._post_processed_model = None

    def _validate_fairness_metric(self):
        if not isinstance(self._fairness_metric, FairnessMetric):
            raise TypeError("The fairness metric is expected to be of type FairnessMetric.")
        if not type(self._fairness_metric) in [DemographicParity, EqualizedOdds]:
            raise ValueError("Currently only DemographicParity and EqualizedOdds are supported "
                             "fairness metrics.")

    def _validate_model(self):
        predict_function = getattr(self._fairness_unaware_model, "predict", None)
        if not predict_function:
            raise ValueError("The model is expected to have a 'predict' method.")
        if not callable(predict_function):
            raise ValueError("The provided model does not have a callable 'predict' method.")

    def _validate_estimator(self):
        fit_function = getattr(self._fairness_unaware_estimator, "fit", None)
        predict_function = getattr(self._fairness_unaware_estimator, "predict", None)
        if not predict_function or not fit_function:
            raise ValueError("The model is expected to have 'fit' and 'predict' methods.")
        if not callable(predict_function) or not callable(fit_function):
            raise ValueError("The provided model does not have callable 'fit' or 'predict' methods.")

    def fit(self, X, y, protected_attribute):
        self._validate_fairness_metric()

        # TODO validate shape of X, y, protected_attribute

        if self._fairness_unaware_estimator:
            # train estimator on data first
            self._validate_estimator()
            self._fairness_unaware_estimator.fit(X, y)
            self._fairness_unaware_model = self._fairness_unaware_estimator

        self._validate_model()

        scores = self._fairness_unaware_model.predict(X)
        roc_curve_based_post_processing_method = None
        if isinstance(self._fairness_metric, DemographicParity):
            roc_curve_based_post_processing_method = \
                _roc_curve_based_post_processing_demographic_parity
        else:
            roc_curve_based_post_processing_method = \
                _roc_curve_based_post_processing_equalized_odds

        self._post_processed_model = roc_curve_based_post_processing_method(
            protected_attribute, y, scores, self._gridsize, self._flip, self._plot)

    def predict(self, X, protected_attribute):
        self._validate_post_processed_model_is_fitted()
        return self._post_processed_model(protected_attribute,
                                          self._fairness_unaware_model.predict(X))
    
    def predict_proba(self, X, protected_attribute):
        self._validate_post_processed_model_is_fitted()
        #TODO
        raise NotImplementedError()
    
    def posterior_proba(self, X, protected_attribute):
        self._validate_post_processed_model_is_fitted()
        #TODO
        raise NotImplementedError()
    
    def posterior_predict_proba(self, X, protected_attribute):
        self._validate_post_processed_model_is_fitted()
        #TODO
        raise NotImplementedError()

    def _validate_post_processed_model_is_fitted(self):
        if not self._post_processed_model:
            raise NotFittedException("It is required to call 'fit' before 'predict'.")


def _roc_curve_based_post_processing_demographic_parity(attributes, labels, scores, gridsize=1000,
                                                        flip=True, plot=False):
    """ Calculates selection and error rates for every attribute value at different thresholds
    over the scores. Subsequently weighs each attribute value's error by the frequency of the
    attribute value in the data. The minimum error point is the selected solution, which is
    recreated by interpolating between two points on the convex hull of all solutions. Each
    attribute value has its own model in the resulting post-processed model, which requires
    the attribute value as an input.

    :param attributes: the protected attributes
    :type attributes: list
    :param labels: the labels of the dataset
    :type labels: list
    :param scores: the scores produced by a model's prediction
    :type scores: list
    :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    :type gridsize: int
    :param flip: allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param plot: show ROC plot if True
    :type plot: bool
    :return: the post-processed model as a function taking the protected attribute value
        and the fairness unaware model's score as arguments to produce predictions
    """
    n = len(labels)
    selection_error_curve = {}
    x_grid = np.linspace(0, 1, gridsize + 1)
    error_given_selection = 0 * x_grid

    data_grouped_by_attribute = _sanity_check_and_group_data(attributes, labels, scores)

    for attribute, group in data_grouped_by_attribute:
        # determine probability of current protected attribute group based on data
        n_group = len(group)
        n_positive = sum(group[LABEL_KEY])
        n_negative = n_group - n_positive
        p_attribute = n_group / n

        roc_convex_hull = _get_roc(group, x_grid, attribute, flip=flip)

        fraction_negative_label_positive_sample = (n_negative / n_group) * roc_convex_hull['x']
        fraction_positive_label_positive_sample = (n_positive / n_group) * roc_convex_hull['y']
        # Calculate selection to represent the proportion of positive predictions.
        roc_convex_hull['selection'] = fraction_negative_label_positive_sample + \
            fraction_positive_label_positive_sample

        fraction_positive_label_negative_sample = \
            (n_positive / n_group) * (1 - roc_convex_hull['y'])
        roc_convex_hull['error'] = fraction_negative_label_positive_sample + \
            fraction_positive_label_negative_sample

        selection_error_curve[attribute] = _interpolate_curve(roc_convex_hull, 'selection',
                                                              'error', 'operation', x_grid)

        # Add up errors for the current group multiplied by the probability of the current group.
        # This will help us in identifying the minimum overall error.
        error_given_selection += p_attribute * selection_error_curve[attribute]['error']

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("Processing " + str(attribute))
        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("DATA")
        logger.debug(group)
        logger.debug("ROC curve: convex")
        logger.debug(roc_convex_hull)
        if plot:
            plot_curve(attribute, 'selection', 'error', selection_error_curve[attribute])

    # Find minimum error point given that at each point the selection rate for each attribute
    # value is identical by design.
    i_best_DP = error_given_selection.idxmin()
    x_best = x_grid[i_best_DP]

    # create the solution as interpolation of multiple points with a separate predictor per
    # protected attribute
    predicted_DP_by_attribute = {}
    for attribute in selection_error_curve.keys():
        # For DP we already have the predictor directly without complex interpolation.
        roc_result = selection_error_curve[attribute].transpose()[i_best_DP]
        predicted_DP_by_attribute[attribute] = _interpolate_prediction(0, 0,
                                                                       roc_result.p0,
                                                                       roc_result.operation0,
                                                                       roc_result.p1,
                                                                       roc_result.operation1)

    logger.debug(OUTPUT_SEPARATOR)
    logger.debug("From ROC curves")
    logger.debug("Best DP: error={0:.3f}, selection rate={1:.3f}"
                 .format(error_given_selection[i_best_DP], x_best))
    logger.debug(OUTPUT_SEPARATOR)
    if plot:
        plot_solution_and_show_plot(x_best, None, "DP solution")

    return lambda A, scores: _vectorized_prediction(predicted_DP_by_attribute, A, scores)


def _roc_curve_based_post_processing_equalized_odds(attributes, labels, scores, gridsize=1000,
                                                    flip=True, plot=False):
    """ Calculates the ROC curve of every attribute value at different thresholds over the scores.
    Subsequently takes the overlapping region of the ROC curves, and finds the best solution by
    selecting the point on the curve with minimal error.

    :param attributes: the protected attributes
    :type attributes: list
    :param labels: the labels of the dataset
    :type labels: list
    :param scores: the scores produced by a model's prediction
    :type scores: list
    :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    :type gridsize: int
    :param flip: allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param plot: show ROC plot if True
    :type plot: bool
    :return: the post-processed model as a function taking the protected attribute value
        and the fairness unaware model's score as arguments to produce predictions
    """
    n = len(labels)
    n_positive = sum(labels)
    n_negative = n - n_positive
    roc = {}
    x_grid = np.linspace(0, 1, gridsize + 1)
    y_values = pd.DataFrame()

    data_grouped_by_attribute = _sanity_check_and_group_data(attributes, labels, scores)

    for attribute, group in data_grouped_by_attribute:
        roc_convex_hull = _get_roc(group, x_grid, attribute, flip=flip)
        roc[attribute] = _interpolate_curve(roc_convex_hull, 'x', 'y', 'operation', x_grid)
        y_values[attribute] = roc[attribute]['y']

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("Processing " + str(attribute))
        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("DATA")
        logger.debug(group)
        logger.debug("ROC curve: convex")
        logger.debug(roc_convex_hull)
        if plot:
            plot_curve(attribute, 'x', 'y', roc[attribute])

    # Calculate the overlap of the ROC curves by taking the lowest y value
    # at every given x.
    y_min = np.amin(y_values, axis=1)
    # Calculate the error at any given x as the sum of
    # a) the proportion of negative labels multiplied by x which represents
    #    the conditional probability P[Y_hat=1 | Y=0], i.e. the probability
    #    of a positive prediction given a negative label.
    # b) the propotion of positive labels multiplied by 1-y_min, where y_min
    #    represents the conditional probability P[Y_hat=1 | Y=1], i.e. the
    #    probability of a correct prediction of a positive label, so 1-y_min
    #    represents a negative prediction given a positive label.
    error_given_x = (n_negative / n) * x_grid + (n_positive / n) * (1 - y_min)
    i_best_EO = error_given_x.idxmin()
    x_best = x_grid[i_best_EO]
    y_best = y_min[i_best_EO]

    # create the solution as interpolation of multiple points with a separate predictor
    # per protected attribute
    predicted_EO_by_attribute = {}
    for attribute in roc.keys():
        roc_result = roc[attribute].transpose()[i_best_EO]
        # p_ignore * x_best represent the diagonal of the ROC plot.
        if roc_result.y == roc_result.x:
            # result is on the diagonal of the ROC plot, i.e. p_ignore is not required
            p_ignore = 0
        else:
            # Calculate p_ignore to change prediction P to y_best
            # p_ignore * x_best + (1 - p_ignore) * P
            difference_from_best_predictor_for_attribute = roc_result.y - y_best
            vertical_distance_from_diagonal = roc_result.y - roc_result.x
            p_ignore = difference_from_best_predictor_for_attribute / \
                vertical_distance_from_diagonal

        predicted_EO_by_attribute[attribute] = _interpolate_prediction(p_ignore, x_best,
                                                                       roc_result.p0,
                                                                       roc_result.operation0,
                                                                       roc_result.p1,
                                                                       roc_result.operation1)

    logger.debug(OUTPUT_SEPARATOR)
    logger.debug("From ROC curves")
    logger.debug("Best EO: error={0:.3f}, FP rate={1:.3f}, TP rate={2:.3f}"
                 .format(error_given_x[i_best_EO], x_best, y_best))
    logger.debug(OUTPUT_SEPARATOR)
    if plot:
        plot_overlap(x_grid, y_min)
        plot_solution_and_show_plot(x_best, y_best, 'EO solution')

    return lambda A, scores: _vectorized_prediction(predicted_EO_by_attribute, A, scores)


def _vectorized_prediction(function_dict, A, scores):
    """ Make predictions for all samples with all provided functions,
    but use only the results from the function that corresponds to the protected
    attribute value of the sample.

    :param function_dict: the functions that apply to various protected attribute values
    :type function_dict: dictionary of functions
    :param A: protected attributes for each sample
    :type A: vector as Series or ndarray
    :param scores: vector of predicted values
    :type scores: vector as Series or ndarray
    """
    A_vector = A
    scores_vector = scores

    if type(A_vector) == list:
        A_vector = np.array(A)

    if type(scores_vector) == list:
        scores_vector = np.array(scores)

    if len(A_vector) != len(scores_vector):
        raise ValueError("The protected attribute vector needs to be of the same length as the "
                         "scores vector.")

    return sum([(A_vector == a) * function_dict[a](scores_vector) for a in function_dict])


def _interpolate_prediction(p_ignore, prediction_constant, p0, operation0, p1, operation1):
    """ Creates the interpolated prediction between two predictions. The predictions
    are represented through the threshold rules operation0 and operation1.

    :param p_ignore: p_ignore changes the interpolated prediction P to the desired
        solution using the transformation p_ignore * prediction_constant + (1 - p_ignore) * P
    :param prediction_constant: 0 if not required, otherwise the x value of the best
        solution should be passed
    :param p0: interpolation multiplier for prediction from the first predictor
    :param operation0: threshold rule for the first predictor
    :param p1: interpolation multiplier for prediction from the second predictor
    :param operation1: threshold rule for the second predictor
    :return: an anonymous function that scales the original prediction to the desired one
    """
    pred0 = operation0.get_predictor_from_operation()
    pred1 = operation1.get_predictor_from_operation()

    logger.debug(OUTPUT_SEPARATOR)
    logger.debug("p_ignore: {}".format(p_ignore))
    logger.debug("prediction_constant: {}".format(prediction_constant))
    logger.debug("p0: {}".format(p0))
    logger.debug("operation0: {}".format(operation0))
    logger.debug("p1: {}".format(p1))
    logger.debug("operation1: {}".format(operation1))
    logger.debug(OUTPUT_SEPARATOR)
    return (lambda scores: p_ignore * prediction_constant +
            (1 - p_ignore) * (p0 * pred0(scores) + p1 * pred1(scores)))


def _sanity_check_and_group_data(attributes, labels, scores):
    # TODO check types: ndarray? list? add test cases as well.
    if len(attributes) == 0 or len(labels) == 0 or len(scores) == 0:
        raise ValueError(EMPTY_INPUT_ERROR_MESSAGE)

    if len(attributes) != len(labels) or len(attributes) != len(scores):
        raise ValueError(DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE)

    if set(np.unique(labels)) > set([0, 1]):
        raise ValueError(NON_BINARY_LABELS_ERROR_MESSAGE)

    data = pd.DataFrame({ATTRIBUTE_KEY: attributes, SCORE_KEY: scores, LABEL_KEY: labels})

    return data.groupby(ATTRIBUTE_KEY)
