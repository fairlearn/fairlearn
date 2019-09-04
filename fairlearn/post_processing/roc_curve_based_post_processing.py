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

from ._constants import LABEL_KEY, SCORE_KEY, ATTRIBUTE_KEY
from ._roc_curve_utilities import _interpolate_curve, _get_roc
from ._roc_curve_plotting_utilities import plot_solution_and_show_plot, plot_overlap, plot_curve

OUTPUT_SEPARATOR = "-"*65

DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "Attributes, labels, and scores need to be of equal length."
EMPTY_INPUT_ERROR_MESSAGE = "At least one of attributes, labels, or scores are empty."
NON_BINARY_LABELS_ERROR_MESSAGE = "Labels other than 0/1 were provided."

logger = logging.getLogger(__name__)


def roc_curve_based_post_processing_demographic_parity(attributes, labels, scores, gridsize=1000,
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
        and the usual input data (x) as arguments to produce predictions
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

    return lambda A, X: _vectorized_prediction(predicted_DP_by_attribute, A, X)


def roc_curve_based_post_processing_equalized_odds(attributes, labels, scores, gridsize=1000,
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
        and the usual input data (x) as arguments to produce predictions
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

    return lambda A, X: _vectorized_prediction(predicted_EO_by_attribute, A, X)


def _vectorized_prediction(function_dict, A, X):
    """ Make predictions for all samples in X with all provided functions,
    but use only the results from the function that corresponds to the protected
    attribute value of the sample.

    :param function_dict: the functions that apply to various protected attribute values
    :type function_dict: dictionary of functions
    :param A: protected attributes for each sample
    :type A: vector as Series or ndarray
    :param X: matrix of samples
    :type X: matrix as DataFrame or ndarray
    """
    return sum([(A == a) * function_dict[a](X) for a in function_dict])


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
    return (lambda X: p_ignore * prediction_constant +
            (1 - p_ignore) * (p0 * pred0(X) + p1 * pred1(X)))


def _sanity_check_and_group_data(attributes, labels, scores):
    # TODO check types: dataframe? ndarray? list? add test cases as well.
    if len(attributes) == 0 or len(labels) == 0 or len(scores) == 0:
        raise ValueError(EMPTY_INPUT_ERROR_MESSAGE)

    if len(attributes) != len(labels) or len(attributes) != len(scores):
        raise ValueError(DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE)

    if set(np.unique(labels)) > set([0, 1]):
        raise ValueError(NON_BINARY_LABELS_ERROR_MESSAGE)

    data = pd.DataFrame({ATTRIBUTE_KEY: attributes, SCORE_KEY: scores, LABEL_KEY: labels})

    return data.groupby(ATTRIBUTE_KEY)
