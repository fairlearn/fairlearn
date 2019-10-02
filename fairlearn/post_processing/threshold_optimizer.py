# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Threshold Optimization Post Processing algorithm based on M. Hardt,
E. Price, N. Srebro's paper "Equality of Opportunity in Supervised
Learning" (https://arxiv.org/pdf/1610.02413.pdf) for binary
classification with one categorical grouping attribute.
"""

import logging
import numpy as np
import pandas as pd
import random

from fairlearn.exceptions import NotFittedException
from fairlearn.post_processing import PostProcessing
from ._constants import (LABEL_KEY, SCORE_KEY, ATTRIBUTE_KEY, OUTPUT_SEPARATOR,
                         DEMOGRAPHIC_PARITY, EQUALIZED_ODDS)
from ._roc_curve_utilities import _interpolate_curve, _get_roc
from ._curve_plotting_utilities import plot_solution_and_show_plot, plot_overlap, plot_curve
from ._interpolated_prediction import InterpolatedPredictor

DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "{} need to be of equal length."
EMPTY_INPUT_ERROR_MESSAGE = "At least one of attributes, labels, or scores are empty."
NON_BINARY_LABELS_ERROR_MESSAGE = "Labels other than 0/1 were provided."
INPUT_DATA_FORMAT_ERROR_MESSAGE = "The only allowed input data formats are: " \
                                  "list, numpy.ndarray, pandas.DataFrame, pandas.Series. " \
                                  "Your provided data was of types ({}, {}, {})"
NOT_SUPPORTED_PARITY_CRITERIA_ERROR_MESSAGE = "Currently only {} and {} are supported " \
    "parity criteria.".format(DEMOGRAPHIC_PARITY, EQUALIZED_ODDS)
PREDICT_BEFORE_FIT_ERROR_MESSAGE = "It is required to call 'fit' before 'predict'."
MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE = "Post processing currently only supports a single " \
    "column in {}."
ATTRIBUTE_NAME_CONFLICT_DETECTED_ERROR_MESSAGE = "An attribute named {} or {} was detected. " \
    "Please rename your column and try again.".format(SCORE_KEY, LABEL_KEY)
SCORES_DATA_TOO_MANY_COLUMNS_ERROR_MESSAGE = "The provided scores data contains multiple columns."
UNEXPECTED_DATA_TYPE_ERROR_MESSAGE = "Unexpected data type {} encountered."


_SUPPORTED_PARITY_CRITERIA = [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS]

logger = logging.getLogger(__name__)


class ThresholdOptimizer(PostProcessing):
    def __init__(self, *, unconstrained_model=None, unconstrained_estimator=None,
                 parity_criteria=DEMOGRAPHIC_PARITY, gridsize=1000, flip=True, plot=False,
                 seed=None):
        """ Creates the post processing object.

        :param unconstrained_model: the trained model whose output will be post processed
        :type unconstrained_model: a trained model
        :param unconstrained_estimator: an untrained estimator that will be trained, and
            subsequently its output will be post processed
        :type unconstrained_estimator: an untrained estimator
        :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
        :type gridsize: int
        :param flip: allow flipping to negative weights if it improves accuracy.
        :type flip: bool
        :param plot: show ROC/selection-error plot if True
        :type plot: bool
        """
        super(ThresholdOptimizer, self).__init__(
            unconstrained_model=unconstrained_model,
            unconstrained_estimator=unconstrained_estimator,
            parity_criteria=parity_criteria)

        self._parity_criteria = parity_criteria
        if self._parity_criteria not in _SUPPORTED_PARITY_CRITERIA:
            raise ValueError(NOT_SUPPORTED_PARITY_CRITERIA_ERROR_MESSAGE)

        self._gridsize = gridsize
        self._flip = flip
        self._plot = plot
        random.seed(seed)
        self._post_processed_model_by_attribute = None

    def fit(self, X, y, aux_data, **kwargs):
        self._validate_input_data(X, aux_data, y)

        if self._unconstrained_estimator:
            # train estimator on data first
            self._validate_estimator()
            self._unconstrained_estimator.fit(X, y, **kwargs)
            self._unconstrained_model = self._unconstrained_estimator

        self._validate_model()

        scores = self._unconstrained_model.predict(X)
        threshold_optimization_method = None
        if self._parity_criteria == DEMOGRAPHIC_PARITY:
            threshold_optimization_method = \
                _threshold_optimization_demographic_parity
        elif self._parity_criteria == EQUALIZED_ODDS:
            threshold_optimization_method = \
                _threshold_optimization_equalized_odds
        else:
            raise ValueError(NOT_SUPPORTED_PARITY_CRITERIA_ERROR_MESSAGE)

        self._post_processed_model_by_attribute = threshold_optimization_method(
            aux_data, y, scores, self._gridsize, self._flip, self._plot)

    def predict(self, X, group_data):
        self._validate_post_processed_model_is_fitted()
        self._validate_input_data(X, group_data)
        unconstrained_predictions = self._unconstrained_model.predict(X)

        positive_probs = _vectorized_prediction(self._post_processed_model_by_attribute,
                                                group_data,
                                                unconstrained_predictions)
        return (positive_probs >= np.random.rand(len(positive_probs))) * 1

    def predict_proba(self, X, group_data):
        self._validate_post_processed_model_is_fitted()
        self._validate_input_data(X, group_data)
        positive_probs = _vectorized_prediction(self._post_processed_model_by_attribute,
                                                group_data,
                                                self._unconstrained_model.predict(X))
        return np.array([[1.0 - p, p] for p in positive_probs])

    def _validate_post_processed_model_is_fitted(self):
        if not self._post_processed_model_by_attribute:
            raise NotFittedException(PREDICT_BEFORE_FIT_ERROR_MESSAGE)

    def _validate_input_data(self, X, aux_data, y=None):
        allowed_input_types = [list, np.ndarray, pd.DataFrame, pd.Series]
        if type(X) not in allowed_input_types or \
                type(aux_data) not in allowed_input_types or \
                (y is not None and type(y) not in allowed_input_types):
            raise TypeError(INPUT_DATA_FORMAT_ERROR_MESSAGE
                            .format(type(X).__name__,
                                    type(y).__name__,
                                    type(aux_data).__name__))

        if len(X) == 0 or len(aux_data) == 0 or (y is not None and len(y) == 0):
            raise ValueError(EMPTY_INPUT_ERROR_MESSAGE)

        if y is None:
            if len(X) != len(aux_data) or (y is not None and len(X) != len(y)):
                raise ValueError(DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE
                                 .format("X and aux_data"))
        else:
            if len(X) != len(aux_data) or (y is not None and len(X) != len(y)):
                raise ValueError(DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE
                                 .format("X, aux_data, and y"))

        if set(np.unique(y)) > set([0, 1]):
            raise ValueError(NON_BINARY_LABELS_ERROR_MESSAGE)


def _threshold_optimization_demographic_parity(attributes, labels, scores, gridsize=1000,
                                               flip=True, plot=False):
    """ Calculates selection and error rates for every attribute value at different thresholds
    over the scores. Subsequently weighs each attribute value's error by the frequency of the
    attribute value in the data. The minimum error point is the selected solution, which is
    recreated by interpolating between two points on the convex hull of all solutions. Each
    attribute value has its own model in the resulting post-processed model, which requires
    the attribute value as an input.

    This method assumes that attributes, labels, and scores are non-empty data structures of
    equal length, and labels contains only binary labels 0 and 1.

    :param attributes: the grouping attribute data
    :type attributes: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param labels: the labels of the dataset
    :type labels: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param scores: the scores produced by a model's prediction
    :type scores: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    :type gridsize: int
    :param flip: allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param plot: show selection-error plot if True
    :type plot: bool
    :return: the post-processed model as a function taking the grouping attribute value
        and the fairness unaware model's score as arguments to produce predictions
    """
    n = len(labels)
    selection_error_curve = {}
    x_grid = np.linspace(0, 1, gridsize + 1)
    error_given_selection = 0 * x_grid

    data_grouped_by_attribute = _reformat_and_group_data(
        attributes, labels, scores)

    for attribute, group in data_grouped_by_attribute:
        # determine probability of current grouping attribute group based on data
        n_group = len(group)
        n_positive = sum(group[LABEL_KEY])
        n_negative = n_group - n_positive
        p_attribute = n_group / n

        roc_convex_hull = _get_roc(group, x_grid, attribute, flip=flip)

        fraction_negative_label_positive_sample = (
            n_negative / n_group) * roc_convex_hull['x']
        fraction_positive_label_positive_sample = (
            n_positive / n_group) * roc_convex_hull['y']
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
        error_given_selection += p_attribute * \
            selection_error_curve[attribute]['error']

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("Processing " + str(attribute))
        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("DATA")
        logger.debug(group)
        logger.debug("ROC curve: convex")
        logger.debug(roc_convex_hull)
        if plot:
            plot_curve(attribute, 'selection', 'error',
                       selection_error_curve[attribute])

    # Find minimum error point given that at each point the selection rate for each attribute
    # value is identical by design.
    i_best_DP = error_given_selection.idxmin()
    x_best = x_grid[i_best_DP]

    # create the solution as interpolation of multiple points with a separate predictor per
    # grouping attribute
    predicted_DP_by_attribute = {}
    for attribute in selection_error_curve.keys():
        # For DP we already have the predictor directly without complex interpolation.
        selection_error_curve_result = selection_error_curve[attribute].transpose()[
            i_best_DP]
        predicted_DP_by_attribute[attribute] = \
            InterpolatedPredictor(0, 0,
                                  selection_error_curve_result.p0,
                                  selection_error_curve_result.operation0,
                                  selection_error_curve_result.p1,
                                  selection_error_curve_result.operation1)

    logger.debug(OUTPUT_SEPARATOR)
    logger.debug("From ROC curves")
    logger.debug("Best DP: error={0:.3f}, selection rate={1:.3f}"
                 .format(error_given_selection[i_best_DP], x_best))
    logger.debug(OUTPUT_SEPARATOR)
    if plot:
        plot_solution_and_show_plot(
            x_best, None, "DP solution", "selection rate", "error")

    return predicted_DP_by_attribute


def _threshold_optimization_equalized_odds(attributes, labels, scores, gridsize=1000,
                                           flip=True, plot=False):
    """ Calculates the ROC curve of every attribute value at different thresholds over the scores.
    Subsequently takes the overlapping region of the ROC curves, and finds the best solution by
    selecting the point on the curve with minimal error.

    This method assumes that attributes, labels, and scores are non-empty data structures of
    equal length, and labels contains only binary labels 0 and 1.

    :param attributes: the grouping attribute data
    :type attributes: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param labels: the labels of the dataset
    :type labels: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param scores: the scores produced by a model's prediction
    :type scores: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    :type gridsize: int
    :param flip: allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param plot: show ROC plot if True
    :type plot: bool
    :return: the post-processed model as a function taking the grouping attribute value
        and the fairness unaware model's score as arguments to produce predictions
    """
    data_grouped_by_attribute = _reformat_and_group_data(
        attributes, labels, scores)

    n = len(labels)

    if type(labels) == pd.DataFrame:
        n_positive = labels.sum().loc[0]
    else:
        n_positive = sum(labels)
    n_negative = n - n_positive
    roc = {}
    x_grid = np.linspace(0, 1, gridsize + 1)
    y_values = pd.DataFrame()

    for attribute, group in data_grouped_by_attribute:
        roc_convex_hull = _get_roc(group, x_grid, attribute, flip=flip)
        roc[attribute] = _interpolate_curve(
            roc_convex_hull, 'x', 'y', 'operation', x_grid)
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
    # per grouping attribute
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

        predicted_EO_by_attribute[attribute] = InterpolatedPredictor(p_ignore, x_best,
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
        plot_solution_and_show_plot(x_best, y_best, 'EO solution', "$P[\\hat{Y}=1|Y=0]$",
                                    "$P[\\hat{Y}=1|Y=1]$")

    return predicted_EO_by_attribute


def _vectorized_prediction(function_dict, group_data, scores):
    """ Make predictions for all samples with all provided functions,
    but use only the results from the function that corresponds to the grouping
    attribute value of the sample.

    This method assumes that A and scores are of equal length.

    :param function_dict: the functions that apply to various grouping attribute values
    :type function_dict: dictionary of functions
    :param group_data: grouping attributes for each sample
    :type group_data: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param scores: vector of predicted values
    :type scores: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    """
    # handle type conversion to ndarray for other types
    group_data_vector = _convert_to_ndarray(group_data, MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE
                                            .format("group_data"))
    scores_vector = _convert_to_ndarray(
        scores, SCORES_DATA_TOO_MANY_COLUMNS_ERROR_MESSAGE)

    return sum([(group_data_vector == a) * function_dict[a].predict(scores_vector)
                for a in function_dict])


def _convert_to_ndarray(data, dataframe_multiple_columns_error_message):
    if type(data) == list:
        data = np.array(data)
    elif type(data) == pd.DataFrame:
        if len(data.columns) > 1:
            # TODO: extend to multiple columns for additional group data
            raise ValueError(dataframe_multiple_columns_error_message)
        data = data[data.columns[0]].values
    elif type(data) == pd.Series:
        data = data.values
    return data


def _reformat_and_group_data(attributes, labels, scores, attribute_names=None):
    data_dict = {}

    # TODO: extend to multiple columns for additional group data
    # and name columns after original column names if possible
    # or store the original column names
    attribute_name = ATTRIBUTE_KEY
    if attribute_names is not None:
        if attribute_name in [SCORE_KEY, LABEL_KEY]:
            raise ValueError(ATTRIBUTE_NAME_CONFLICT_DETECTED_ERROR_MESSAGE)
        attribute_name = attribute_names[0]

    _reformat_data_into_dict(attribute_name, data_dict, attributes)
    _reformat_data_into_dict(SCORE_KEY, data_dict, scores)
    _reformat_data_into_dict(LABEL_KEY, data_dict, labels)

    return pd.DataFrame(data_dict).groupby(attribute_name)


def _reformat_data_into_dict(key, data_dict, additional_data):
    if type(additional_data) == np.ndarray:
        if len(additional_data.shape) > 2 or (len(additional_data.shape) == 2 and
                                              additional_data.shape[1] > 1):
            # TODO: extend to multiple columns for additional_group data
            raise ValueError(
                MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE.format("aux_data"))
        else:
            data_dict[key] = additional_data.reshape(-1)
    elif type(additional_data) == pd.DataFrame:
        # TODO: extend to multiple columns for additional_data by using attribute_column
        for attribute_column in additional_data.columns:
            data_dict[key] = additional_data[attribute_column].values
    elif type(additional_data) == pd.Series:
        data_dict[key] = additional_data.values
    elif type(additional_data) == list:
        if type(additional_data[0]) == list:
            if len(additional_data[0]) > 1:
                # TODO: extend to multiple columns for additional_data
                raise ValueError(
                    MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE.format("aux_data"))
            data_dict[key] = map(lambda a: a[0], additional_data)
        else:
            data_dict[key] = additional_data
    else:
        raise TypeError(UNEXPECTED_DATA_TYPE_ERROR_MESSAGE.format(
            type(additional_data)))
