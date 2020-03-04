# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Threshold Optimization Post Processing algorithm.

This is based on M. Hardt, E. Price, N. Srebro's paper
"`Equality of Opportunity in Supervised Learning
<https://arxiv.org/pdf/1610.02413.pdf>`_" for binary
classification with one categorical sensitive feature.
"""

import logging
import numpy as np
import pandas as pd
import random

from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from fairlearn._input_validation import _validate_and_reformat_input
from ._constants import (LABEL_KEY, SCORE_KEY, SENSITIVE_FEATURE_KEY, OUTPUT_SEPARATOR,
                         DEMOGRAPHIC_PARITY, EQUALIZED_ODDS)
from ._roc_curve_utilities import _interpolate_curve, _get_roc
from ._interpolated_prediction import InterpolatedPredictor

# various error messages
DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "{} need to be of equal length."
NON_BINARY_LABELS_ERROR_MESSAGE = "Labels other than 0/1 were provided."
NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE = "Currently only {} and {} are supported " \
    "constraints.".format(DEMOGRAPHIC_PARITY, EQUALIZED_ODDS)
MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE = "Post processing currently only supports a single " \
    "column in {}."
SENSITIVE_FEATURE_NAME_CONFLICT_DETECTED_ERROR_MESSAGE = "A sensitive feature named {} or {} " \
    "was detected. Please rename your column and try again.".format(SCORE_KEY, LABEL_KEY)
SCORES_DATA_TOO_MANY_COLUMNS_ERROR_MESSAGE = "The provided scores data contains multiple columns."
UNEXPECTED_DATA_TYPE_ERROR_MESSAGE = "Unexpected data type {} encountered."
ESTIMATOR_ERROR_MESSAGE = "'estimator' cannot be `None`."

_SUPPORTED_CONSTRAINTS = [DEMOGRAPHIC_PARITY, EQUALIZED_ODDS]

logger = logging.getLogger(__name__)


class ThresholdOptimizer(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """An Estimator based on the threshold optimization approach.

    The procedure followed is described in detail in
    `Hardt et al. (2016) <https://arxiv.org/abs/1610.02413>`_.

    :param estimator: An estimator whose output will be post processed
    :type estimator: An estimator
    :param grid_size: The number of ticks on the grid over which we evaluate the curves.
        A large grid_size means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    :type grid_size: int
    :param flip: Allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param prefit: If ``True``, avoid re-fitting the given estimator if it's
        already trained. Note that when used with ``cross_val_score``,
        ``GridSearchCV`` and similar utilities that clone the estimator,
        the effective behavior is ``prefit=False``.
    :type: bool, default=False
    """

    def __init__(self, *, estimator=None,
                 constraints=DEMOGRAPHIC_PARITY, grid_size=1000, flip=True,
                 prefit=False):
        self.grid_size = grid_size
        self.flip = flip
        self.post_processed_predictor_by_sensitive_feature = None
        self.constraints = constraints
        self.estimator = estimator
        self.prefit = prefit

    def fit(self, X, y, *, sensitive_features, **kwargs):
        """Fit the model.

        The fit is based on training features and labels, sensitive features,
        as well as the fairness-unaware predictor or estimator. If an estimator was passed
        in the constructor this fit method will call `fit(X, y, **kwargs)` on said estimator.

        :param X: The feature matrix
        :type X: numpy.ndarray or pandas.DataFrame
        :param y: The label vector
        :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
        :param sensitive_features: sensitive features to identify groups by, currently allows
            only a single column
        :type sensitive_features: currently 1D array as numpy.ndarray, list, pandas.DataFrame,
            or pandas.Series
        """
        if self.estimator is None:
            raise ValueError(ESTIMATOR_ERROR_MESSAGE)

        if self.constraints not in _SUPPORTED_CONSTRAINTS:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        _, _, sensitive_feature_vector = _validate_and_reformat_input(
            X, y, sensitive_features=sensitive_features, enforce_binary_labels=True)

        # postprocessing can't handle 0/1 as floating point numbers, so this converts it to int
        if type(y) in [np.ndarray, pd.DataFrame, pd.Series]:
            y = y.astype(int)
        else:
            y = [int(y_val) for y_val in y]

        if not self.prefit:
            self.estimator_ = clone(self.estimator).fit(X, y, **kwargs)
        else:
            try:
                check_is_fitted(self.estimator)
                self.estimator_ = self.estimator
            except NotFittedError:
                self.estimator_ = clone(self.estimator).fit(X, y, **kwargs)

        scores = self.estimator_.predict(X)
        threshold_optimization_method = None
        if self.constraints == DEMOGRAPHIC_PARITY:
            threshold_optimization_method = \
                self._threshold_optimization_demographic_parity
        elif self.constraints == EQUALIZED_ODDS:
            threshold_optimization_method = \
                self._threshold_optimization_equalized_odds
        else:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        self._post_processed_predictor_by_sensitive_feature = threshold_optimization_method(
            sensitive_feature_vector, y, scores, self.grid_size, self.flip)

    def predict(self, X, *, sensitive_features, random_state=None):
        """Predict label for each sample in X while taking into account sensitive features.

        :param X: feature matrix
        :type X: numpy.ndarray or pandas.DataFrame
        :param sensitive_features: sensitive features to identify groups by, currently allows
            only a single column
        :type sensitive_features: currently 1D array as numpy.ndarray, list, pandas.DataFrame,
            or pandas.Series
        :param random_state: set to a constant for reproducibility
        :type random_state: int
        :return: predictions in numpy.ndarray
        """
        if random_state:
            random.seed(random_state)

        check_is_fitted(self)
        _, _, sensitive_feature_vector = _validate_and_reformat_input(
            X, y=None, sensitive_features=sensitive_features, expect_y=False,
            enforce_binary_labels=True)
        unconstrained_predictions = self.estimator_.predict(X)

        positive_probs = _vectorized_prediction(
            self._post_processed_predictor_by_sensitive_feature,
            sensitive_feature_vector,
            unconstrained_predictions)
        return (positive_probs >= np.random.rand(len(positive_probs))) * 1

    def _pmf_predict(self, X, *, sensitive_features):
        """Probabilistic mass function.

        :param X: Feature matrix
        :type X: numpy.ndarray or pandas.DataFrame
        :param sensitive_features: Sensitive features to identify groups by, currently allows
            only a single column
        :type sensitive_features: Currently 1D array as numpy.ndarray, list, pandas.DataFrame,
            or pandas.Series
        :return: array of tuples with probabilities for predicting 0 or 1, respectively. The sum
            of the two numbers in each tuple needs to add up to 1.
        :rtype: numpy.ndarray
        """
        check_is_fitted(self)
        _, _, sensitive_feature_vector = _validate_and_reformat_input(
            X, y=None, sensitive_features=sensitive_features, expect_y=False,
            enforce_binary_labels=True)
        positive_probs = _vectorized_prediction(
            self._post_processed_predictor_by_sensitive_feature, sensitive_feature_vector,
            self.estimator_.predict(X))
        return np.array([[1.0 - p, p] for p in positive_probs])

    def _threshold_optimization_demographic_parity(self, sensitive_features, labels, scores,
                                                   grid_size=1000, flip=True):
        """Calculate the selection and error rates for every sensitive feature value.

        These calculations are made at different
        thresholds over the scores. Subsequently weighs each sensitive feature value's error by the
        frequency of the sensitive feature value in the data. The minimum error point is the
        selected solution, which is recreated by interpolating between two points on the convex
        hull of all solutions. Each sensitive feature value has its own predictor in the resulting
        postprocessed predictor, which requires the sensitive feature value as an input.

        This method assumes that sensitive_features, labels, and scores are non-empty data
        structures of equal length, and labels contains only binary labels 0 and 1.

        :param sensitive_features: the feature data that determines the groups for which the parity
            constraints are enforced
        :type sensitive_features: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        :param labels: the labels of the dataset
        :type labels: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        :param scores: the scores produced by a predictor's prediction
        :type scores: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        :param grid_size: The number of ticks on the grid over which we evaluate the curves.
            A large grid_size means that we approximate the actual curve, so it increases the
            chance of being very close to the actual best solution.
        :type grid_size: int
        :param flip: allow flipping to negative weights if it improves accuracy.
        :type flip: bool
        :return: the postprocessed predictor as a function taking the sensitive feature value
            and the fairness unaware predictor's score as arguments to produce predictions
        """
        n = len(labels)
        self._selection_error_curve = {}
        self._x_grid = np.linspace(0, 1, grid_size + 1)
        error_given_selection = 0 * self._x_grid

        data_grouped_by_sensitive_feature = _reformat_and_group_data(
            sensitive_features, labels, scores)

        for sensitive_feature_value, group in data_grouped_by_sensitive_feature:
            # determine probability of current sensitive feature group based on data
            n_group = len(group)
            n_positive = sum(group[LABEL_KEY])
            n_negative = n_group - n_positive
            p_sensitive_feature_value = n_group / n

            roc_convex_hull = _get_roc(group, sensitive_feature_value, flip=flip)

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

            self._selection_error_curve[sensitive_feature_value] = \
                _interpolate_curve(roc_convex_hull, 'selection', 'error', 'operation',
                                   self._x_grid)

            # Add up errors for the current group multiplied by the probability of the current
            # group. This will help us in identifying the minimum overall error.
            error_given_selection += p_sensitive_feature_value * \
                self._selection_error_curve[sensitive_feature_value]['error']

            logger.debug(OUTPUT_SEPARATOR)
            logger.debug("Processing %s", str(sensitive_feature_value))
            logger.debug(OUTPUT_SEPARATOR)
            logger.debug("DATA")
            logger.debug(group)
            logger.debug("ROC curve: convex")
            logger.debug(roc_convex_hull)

        # Find minimum error point given that at each point the selection rate for each sensitive
        # feature value is identical by design.
        i_best_DP = error_given_selection.idxmin()
        self._x_best = self._x_grid[i_best_DP]

        # create the solution as interpolation of multiple points with a separate predictor per
        # sensitive feature value
        predicted_DP_by_sensitive_feature = {}
        for sensitive_feature_value in self._selection_error_curve.keys():
            # For DP we already have the predictor directly without complex interpolation.
            selection_error_curve_result = self._selection_error_curve[sensitive_feature_value] \
                .transpose()[i_best_DP]
            predicted_DP_by_sensitive_feature[sensitive_feature_value] = \
                InterpolatedPredictor(0, 0,
                                      selection_error_curve_result.p0,
                                      selection_error_curve_result.operation0,
                                      selection_error_curve_result.p1,
                                      selection_error_curve_result.operation1)

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("From ROC curves")
        logger.debug("Best DP: error=%.3f, selection rate=%.3f",
                     error_given_selection[i_best_DP], self._x_best)
        logger.debug(OUTPUT_SEPARATOR)

        return predicted_DP_by_sensitive_feature

    def _threshold_optimization_equalized_odds(self, sensitive_features, labels, scores,
                                               grid_size=1000, flip=True):
        """Calculate the ROC curve of every sensitive feature value at different thresholds.

        Subsequently takes the overlapping region of the ROC curves, and finds the best
        solution by selecting the point on the curve with minimal error.

        This method assumes that sensitive_features, labels, and scores are non-empty data
        structures of equal length, and labels contains only binary labels 0 and 1.

        :param sensitive_features: the feature data that determines the groups for which the
            parity constraints are enforced
        :type sensitive_features: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        :param labels: the labels of the dataset
        :type labels: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        :param scores: the scores produced by a predictor's prediction
        :type scores: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        :param grid_size: The number of ticks on the grid over which we evaluate the curves.
            A large grid_size means that we approximate the actual curve, so it increases the
            chance of being very close to the actual best solution.
        :type grid_size: int
        :param flip: allow flipping to negative weights if it improves accuracy.
        :type flip: bool
        :return: the postprocessed predictor as a function taking the sensitive feature value
            and the fairness unaware predictor's score as arguments to produce predictions
        """
        data_grouped_by_sensitive_feature = _reformat_and_group_data(
            sensitive_features, labels, scores)

        n = len(labels)

        if type(labels) == pd.DataFrame:
            n_positive = labels.sum().loc[0]
        else:
            n_positive = sum(labels)
        n_negative = n - n_positive
        self._roc_curve = {}
        self._x_grid = np.linspace(0, 1, grid_size + 1)
        y_values = pd.DataFrame()

        for sensitive_feature_value, group in data_grouped_by_sensitive_feature:
            roc_convex_hull = _get_roc(group, sensitive_feature_value, flip=flip)
            self._roc_curve[sensitive_feature_value] = \
                _interpolate_curve(roc_convex_hull, 'x', 'y', 'operation', self._x_grid)
            y_values[sensitive_feature_value] = self._roc_curve[sensitive_feature_value]['y']

            logger.debug(OUTPUT_SEPARATOR)
            logger.debug("Processing %s", str(sensitive_feature_value))
            logger.debug(OUTPUT_SEPARATOR)
            logger.debug("DATA")
            logger.debug(group)
            logger.debug("ROC curve: convex")
            logger.debug(roc_convex_hull)

        # Calculate the overlap of the ROC curves by taking the lowest y value
        # at every given x.
        self._y_min = np.amin(y_values, axis=1)
        # Calculate the error at any given x as the sum of
        # a) the proportion of negative labels multiplied by x which represents
        #    the conditional probability P[Y_hat=1 | Y=0], i.e. the probability
        #    of a positive prediction given a negative label.
        # b) the propotion of positive labels multiplied by 1-y_min, where y_min
        #    represents the conditional probability P[Y_hat=1 | Y=1], i.e. the
        #    probability of a correct prediction of a positive label, so 1-y_min
        #    represents a negative prediction given a positive label.
        error_given_x = (n_negative / n) * self._x_grid + (n_positive / n) * (1 - self._y_min)
        i_best_EO = error_given_x.idxmin()
        self._x_best = self._x_grid[i_best_EO]
        self._y_best = self._y_min[i_best_EO]

        # create the solution as interpolation of multiple points with a separate predictor
        # per sensitive feature
        predicted_EO_by_sensitive_feature = {}
        for sensitive_feature_value in self._roc_curve.keys():
            roc_result = self._roc_curve[sensitive_feature_value].transpose()[i_best_EO]
            # p_ignore * x_best represent the diagonal of the ROC plot.
            if roc_result.y == roc_result.x:
                # result is on the diagonal of the ROC plot, i.e. p_ignore is not required
                p_ignore = 0
            else:
                # Calculate p_ignore to change prediction P to y_best
                # p_ignore * x_best + (1 - p_ignore) * P
                difference_from_best_predictor_for_sensitive_feature = roc_result.y - self._y_best
                vertical_distance_from_diagonal = roc_result.y - roc_result.x
                p_ignore = difference_from_best_predictor_for_sensitive_feature / \
                    vertical_distance_from_diagonal

            predicted_EO_by_sensitive_feature[sensitive_feature_value] = \
                InterpolatedPredictor(p_ignore, self._x_best,
                                      roc_result.p0, roc_result.operation0,
                                      roc_result.p1, roc_result.operation1)

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("From ROC curves")
        logger.debug("Best EO: error=%.3f}, FP rate=%.3f}, TP rate=%.3f}",
                     error_given_x[i_best_EO], self._x_best, self._y_best)
        logger.debug(OUTPUT_SEPARATOR)

        return predicted_EO_by_sensitive_feature


def _vectorized_prediction(function_dict, sensitive_features, scores):
    """Make predictions for all samples with all provided functions.

    However, only use the results from the function that corresponds to the
    sensitive feature value of the sample.

    This method assumes that sensitive_features and scores are of equal length.

    :param function_dict: the functions that apply to various sensitive feature values
    :type function_dict: dictionary of functions
    :param sensitive_features: the feature data that determines the grouping
    :type sensitive_features: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    :param scores: vector of predicted values
    :type scores: list, numpy.ndarray, pandas.DataFrame, or pandas.Series
    """
    # handle type conversion to ndarray for other types
    sensitive_features_vector = np.array(sensitive_features)
    scores_vector = np.array(scores)

    return sum([(sensitive_features_vector == a) * function_dict[a].predict(scores_vector)
                for a in function_dict])


def _reformat_and_group_data(sensitive_features, labels, scores, sensitive_feature_names=None):
    """Reformats the data into a new pandas.DataFrame and group by sensitive feature values.

    The data are provided as three arguments (`sensitive_features`, `labels`, `scores`) and
    the new  DataFrame is grouped by sensitive feature values so that subsequently each group
    can be handled separately.

    :param sensitive_features: the sensitive features based on which the grouping is determined;
        currently only a single sensitive feature is supported
    :type sensitive_features: pandas.Series, pandas.DataFrame, numpy.ndarray, or list
    :param labels: the training labels
    :type labels: pandas.Series, pandas.DataFrame, numpy.ndarray, or list
    :param scores: the output from the unconstrained predictor used for training the mitigator
    :type scores: pandas.Series, pandas.DataFrame, numpy.ndarray, or list
    :param sensitive_feature_names: list of names for the sensitive features in case they were
        not implicitly provided (e.g. if `sensitive_features` is of type DataFrame); default
        None
    :type sensitive_feature_names: list of strings
    :return: the training data for the mitigator, grouped by sensitive feature value
    :rtype: pandas.DataFrameGroupBy
    """
    data_dict = {}

    # TODO: extend to multiple columns for additional group data
    # and name columns after original column names if possible
    # or store the original column names
    sensitive_feature_name = SENSITIVE_FEATURE_KEY
    if sensitive_feature_names is not None:
        if sensitive_feature_name in [SCORE_KEY, LABEL_KEY]:
            raise ValueError(SENSITIVE_FEATURE_NAME_CONFLICT_DETECTED_ERROR_MESSAGE)
        sensitive_feature_name = sensitive_feature_names[0]

    _reformat_data_into_dict(sensitive_feature_name, data_dict, sensitive_features)
    _reformat_data_into_dict(SCORE_KEY, data_dict, scores)
    _reformat_data_into_dict(LABEL_KEY, data_dict, labels)

    return pd.DataFrame(data_dict).groupby(sensitive_feature_name)


def _reformat_data_into_dict(key, data_dict, additional_data):
    """Add `additional_data` to `data_dict` with key `key`.

    Before `additional_data` is added to `data_dict` it is first
    reformatted into a numpy.ndarray or list.

    :param key: the key in `data_dict` at which `additional_data` should be stored;
        `key` should describe the purpose of `additional_data` in `data_dict`
    :type key: str
    :param data_dict: the dictionary containing all the relevant data; `additional_data` will be
        inserted at the key `key`.
    :type data_dict: dict
    :param additional_data: the data to be added to `data_dict` at the specified `key`
    :type additional_data: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
    """
    if type(additional_data) == np.ndarray:
        if len(additional_data.shape) > 2 or (len(additional_data.shape) == 2 and
                                              additional_data.shape[1] > 1):
            # TODO: extend to multiple columns for additional_group data
            raise ValueError(
                MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE.format("sensitive_features"))
        else:
            data_dict[key] = additional_data.squeeze()
    elif type(additional_data) == pd.DataFrame:
        # TODO: extend to multiple columns for additional_data by using column names
        for attribute_column in additional_data.columns:
            data_dict[key] = additional_data[attribute_column].values
    elif type(additional_data) == pd.Series:
        data_dict[key] = additional_data.values
    elif type(additional_data) == list:
        if type(additional_data[0]) == list:
            if len(additional_data[0]) > 1:
                # TODO: extend to multiple columns for additional_data
                raise ValueError(
                    MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE.format("sensitive_features"))
            data_dict[key] = map(lambda a: a[0], additional_data)
        else:
            data_dict[key] = additional_data
    else:
        raise TypeError(UNEXPECTED_DATA_TYPE_ERROR_MESSAGE.format(
            type(additional_data)))
