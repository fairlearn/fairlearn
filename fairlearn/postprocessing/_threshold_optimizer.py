# Copyright (c) Microsoft Corporation and Fairlearn contributors.
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

from warnings import warn

from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import Bunch

from fairlearn._input_validation import (
    _validate_and_reformat_input,
    _KW_CONTROL_FEATURES)
from ._constants import (
    LABEL_KEY, SCORE_KEY, SENSITIVE_FEATURE_KEY,
    OUTPUT_SEPARATOR,
    BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
    BASE_ESTIMATOR_NOT_FITTED_WARNING)
from ._tradeoff_curve_utilities import (
    _interpolate_curve, _tradeoff_curve,
    _extend_confusion_matrix,
    METRIC_DICT)
from ._interpolated_thresholder import InterpolatedThresholder

# various error messages
DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "{} need to be of equal length."
NON_BINARY_LABELS_ERROR_MESSAGE = "Labels other than 0/1 were provided."
MULTIPLE_DATA_COLUMNS_ERROR_MESSAGE = "Post processing currently only supports a single " \
    "column in {}."
SENSITIVE_FEATURE_NAME_CONFLICT_DETECTED_ERROR_MESSAGE = "A sensitive feature named {} or {} " \
    "was detected. Please rename your column and try again.".format(SCORE_KEY, LABEL_KEY)
SCORES_DATA_TOO_MANY_COLUMNS_ERROR_MESSAGE = "The provided scores data contains multiple columns."
UNEXPECTED_DATA_TYPE_ERROR_MESSAGE = "Unexpected data type {} encountered."

logger = logging.getLogger(__name__)


# Simple constraints are described by metrics with values between 0 and 1,
# which attain both extremes as the threshold goes from -Inf to Inf.
# These metrics are also required to be "moments" in the same sense as
# required by fairlearn.reductions, so that the interpolation is possible.
SIMPLE_CONSTRAINTS = {
    'selection_rate_parity': 'selection_rate',
    'demographic_parity': 'selection_rate',
    'false_positive_rate_parity': 'false_positive_rate',
    'false_negative_rate_parity': 'false_negative_rate',
    'true_positive_rate_parity': 'true_positive_rate',
    'true_negative_rate_parity': 'true_negative_rate',
}

ALL_CONSTRAINTS = list(SIMPLE_CONSTRAINTS.keys()) + ['equalized_odds']

# Any "moment" is allowed as a performance metric for simple constraints.
OBJECTIVES_FOR_SIMPLE_CONSTRAINTS = {
    'selection_rate',
    'true_positive_rate',
    'true_negative_rate',
    'accuracy_score',
    'balanced_accuracy_score',
}

# Besides simple constraints we also allow 'equalized_odds' as a constraint.

# For equalized odds, we only allow objectives that are non-decreasing in true_positives,
# when holding n, positives, negatives, true_negatives, and false_positives fixed.
OBJECTIVES_FOR_EQUALIZED_ODDS = {
    'accuracy_score',
    'balanced_accuracy_score',
}

NO_CONTROL_FEATURES = "Control features are not supported by ThresholdOptimizer"
NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE = (
    "Currently only the following constraints are supported: {}.".format(
        ", ".join(sorted(ALL_CONSTRAINTS))))
NOT_SUPPORTED_OBJECTIVES_FOR_SIMPLE_CONSTRAINTS_ERROR_MESSAGE = (
    "For {{}} only the following objectives are supported: {}.".format(
        ", ".join(sorted(OBJECTIVES_FOR_SIMPLE_CONSTRAINTS))))
NOT_SUPPORTED_OBJECTIVES_FOR_EQUALIZED_ODDS_ERROR_MESSAGE = (
    "For equalized_odds only the following objectives are supported: {}.".format(
        ", ".join(sorted(OBJECTIVES_FOR_EQUALIZED_ODDS))))


class ThresholdOptimizer(BaseEstimator, MetaEstimatorMixin):
    """A classifier based on the threshold optimization approach.

    The classifier is obtained by applying group-specific thresholds to
    the provided estimator. The thresholds are chosen to optimize the
    provided performance objective subject to the provided fairness constraints.

    Parameters
    ----------
    estimator : estimator object implementing 'predict' and possibly 'fit'
        An estimator whose output is postprocessed.

    constraints : str, default='demographic_parity'
        Fairness constraints under which threshold optimization is performed.
        Possible inputs are:

            'demographic_parity', 'selection_rate_parity' (synonymous)
                match the selection rate across groups

            '{false,true}_{positive,negative}_rate_parity'
                match the named metric across groups

            'equalized_odds'
                match true positive and false positive rates across groups

    objective : str, default='accuracy_score'
        Performance objective under which threshold optimization is performed.
        Not all objectives are allowed for all types of constraints.
        Possible inputs are:

            'accuracy_score', 'balanced_accuracy_score'
                allowed for all constraint types

            'selection_rate', 'true_positive_rate', 'true_negative_rate',
                allowed for all constraint types except 'equalized_odds'

    grid_size : int, default=1000
        The values of the constraint metric are discretized according to the grid
        of the specified size over the interval [0,1] and the optimization is
        performed with respect to the constraints achieving those values. In case
        of 'equalized_odds' the constraint metric is the false positive rate.

    flip : bool, default=False
        If True, then allow flipping the decision if it improves the resulting

    prefit : bool, default=False
        If True, avoid refitting the given estimator. Note that when used
        with :func:`sklearn.model_selection.cross_val_score`,
        :class:`sklearn.model_selection.GridSearchCV`, this will result in an error.
        In that case, please use ``prefit=False``.

    Notes
    -----
    The procedure is based on the algorithm of
    `Hardt et al. (2016) <https://arxiv.org/abs/1610.02413>`_.
    """

    def __init__(self, *, estimator=None,
                 constraints="demographic_parity", objective="accuracy_score",
                 grid_size=1000, flip=False, prefit=False):
        self.estimator = estimator
        self.constraints = constraints
        self.objective = objective
        self.grid_size = grid_size
        self.flip = flip
        self.prefit = prefit

    def fit(self, X, y, *, sensitive_features, **kwargs):
        """Fit the model.

        The fit is based on training features and labels, sensitive features,
        as well as the fairness-unaware predictor or estimator. If an estimator was passed
        in the constructor this fit method will call `fit(X, y, **kwargs)` on said estimator.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            The label vector
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, or pandas.Series
            sensitive features to identify groups by
        """
        if self.estimator is None:
            raise ValueError(BASE_ESTIMATOR_NONE_ERROR_MESSAGE)

        if self.constraints in SIMPLE_CONSTRAINTS:
            if self.objective not in OBJECTIVES_FOR_SIMPLE_CONSTRAINTS:
                raise ValueError(
                    NOT_SUPPORTED_OBJECTIVES_FOR_SIMPLE_CONSTRAINTS_ERROR_MESSAGE.format(
                        self.constraints))
        elif self.constraints == "equalized_odds":
            if self.objective not in OBJECTIVES_FOR_EQUALIZED_ODDS:
                raise ValueError(
                    NOT_SUPPORTED_OBJECTIVES_FOR_EQUALIZED_ODDS_ERROR_MESSAGE)
        else:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        if kwargs.get(_KW_CONTROL_FEATURES) is not None:
            raise ValueError(NO_CONTROL_FEATURES)

        _, _, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X, y, sensitive_features=sensitive_features, enforce_binary_labels=True)

        # postprocessing can't handle 0/1 as floating point numbers, so this converts it to int
        if type(y) in [np.ndarray, pd.DataFrame, pd.Series]:
            y = y.astype(int)
        else:
            y = [int(y_val) for y_val in y]

        if not self.prefit:
            # Following is on two lines due to issue when estimator comes from TensorFlow
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **kwargs)
        else:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError:
                warn(BASE_ESTIMATOR_NOT_FITTED_WARNING.format(type(self).__name__))
            self.estimator_ = self.estimator

        scores = self.estimator_.predict(X)
        if self.constraints == "equalized_odds":
            self.x_metric_ = "false_positive_rate"
            self.y_metric_ = "true_positive_rate"
            threshold_optimization_method = self._threshold_optimization_for_equalized_odds
        else:
            self.x_metric_ = SIMPLE_CONSTRAINTS[self.constraints]
            self.y_metric_ = self.objective
            threshold_optimization_method = self._threshold_optimization_for_simple_constraints

        self.interpolated_thresholder_ = threshold_optimization_method(
            sensitive_feature_vector, y, scores)
        return self

    def predict(self, X, *, sensitive_features, random_state=None):
        """Predict label for each sample in X while taking into account sensitive features.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            feature matrix
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, pandas.Series
            sensitive features to identify groups by
        random_state : int or RandomState instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.

        Returns
        -------
        Scalar or vector as numpy.ndarray
            The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        """
        check_is_fitted(self)
        return self.interpolated_thresholder_.predict(
            X, sensitive_features=sensitive_features, random_state=random_state)

    def _pmf_predict(self, X, *, sensitive_features):
        """Probabilistic mass function.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature matrix
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, pandas.Series
            Sensitive features to identify groups by

        Returns
        -------
        numpy.ndarray
            array of tuples with probabilities for predicting 0 or 1,
            respectively. The sum of the two numbers in each tuple needs to
            add up to 1.
        """
        check_is_fitted(self)
        return self.interpolated_thresholder_._pmf_predict(
            X, sensitive_features=sensitive_features)

    def _threshold_optimization_for_simple_constraints(self, sensitive_features, labels, scores):
        """Calculate the objective value across all values of constraints.

        These calculations are made at different thresholds over the scores. Subsequently weighs
        each sensitive feature value's objective by the
        frequency of the sensitive feature value in the data. The maximum objective point is the
        selected solution, which is recreated by interpolating between two points on the convex
        hull of all solutions. Each sensitive feature value has its own predictor in the resulting
        postprocessed predictor, which requires the sensitive feature value as an input.

        This method assumes that sensitive_features, labels, and scores are non-empty data
        structures of equal length, and labels contains only binary labels 0 and 1.

        Parameters
        ----------
        sensitive_features : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
            Sensitive features to identify groups by
        labels : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
            the labels of the dataset
        scores : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
            the scores produced by a predictor's prediction

        Returns
        -------
        InterpolatedThresholder
            the postprocessed predictor.
        """
        n = len(labels)
        self._tradeoff_curve = {}
        self._x_grid = np.linspace(0, 1, self.grid_size + 1)
        overall_tradeoff_curve = 0 * self._x_grid

        data_grouped_by_sensitive_feature = _reformat_and_group_data(
            sensitive_features, labels, scores)

        for sensitive_feature_value, group in data_grouped_by_sensitive_feature:
            # Determine probability of current sensitive feature group based on data.
            p_sensitive_feature_value = len(group) / n

            roc_convex_hull = _tradeoff_curve(
                group, sensitive_feature_value, flip=self.flip,
                x_metric=self.x_metric_, y_metric=self.y_metric_)

            self._tradeoff_curve[sensitive_feature_value] = \
                _interpolate_curve(roc_convex_hull, 'x', 'y', 'operation',
                                   self._x_grid)

            # Add up objective for the current group multiplied by the probability of the current
            # group. This will help us in identifying the maximum overall objective.
            overall_tradeoff_curve += p_sensitive_feature_value * \
                self._tradeoff_curve[sensitive_feature_value]['y']

            logger.debug(OUTPUT_SEPARATOR)
            logger.debug("Processing %s", str(sensitive_feature_value))
            logger.debug(OUTPUT_SEPARATOR)
            logger.debug("DATA")
            logger.debug(group)
            logger.debug("Tradeoff curve")
            logger.debug(roc_convex_hull)

        # Find maximum objective point given that at each point the constraint value for each
        # sensitive feature value is identical by design.
        i_best = overall_tradeoff_curve.idxmax()
        self._x_best = self._x_grid[i_best]

        # Create the solution as interpolation of multiple points with a separate
        # interpolation per sensitive feature value.
        interpolation_dict = {}
        for sensitive_feature_value in self._tradeoff_curve.keys():
            best_interpolation = self._tradeoff_curve[sensitive_feature_value] \
                .transpose()[i_best]
            interpolation_dict[sensitive_feature_value] = \
                Bunch(p0=best_interpolation.p0,
                      operation0=best_interpolation.operation0,
                      p1=best_interpolation.p1,
                      operation1=best_interpolation.operation1)

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("From tradeoff curves")
        logger.debug("Best point (simple constraints): %s=%.3f, %s=%.3f",
                     self.y_metric_, overall_tradeoff_curve[i_best], self.x_metric_, self._x_best)
        logger.debug(OUTPUT_SEPARATOR)

        return InterpolatedThresholder(
            self.estimator_, interpolation_dict, prefit=True).fit(None, None)

    def _threshold_optimization_for_equalized_odds(self, sensitive_features, labels, scores):
        """Calculate the ROC curve of every sensitive feature value at different thresholds.

        Subsequently takes the overlapping region of the ROC curves, and finds the best
        solution by selecting the point on the curve with maximum objective value.

        This method assumes that sensitive_features, labels, and scores are non-empty data
        structures of equal length, and labels contains only binary labels 0 and 1.

        Parameters
        ----------
        sensitive_features : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
            Sensitive features to identify groups by
        labels : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
            the labels of the dataset
        scores : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
            the scores produced by a predictor's prediction

        Returns
        -------
        InterpolatedThresholder
            the postprocessed predictor.
        """
        data_grouped_by_sensitive_feature = _reformat_and_group_data(
            sensitive_features, labels, scores)

        n = len(labels)

        if type(labels) == pd.DataFrame:
            n_positive = labels.sum().loc[0]
        else:
            n_positive = sum(labels)
        n_negative = n - n_positive
        self._tradeoff_curve = {}
        self._x_grid = np.linspace(0, 1, self.grid_size + 1)
        y_values = pd.DataFrame()

        for sensitive_feature_value, group in data_grouped_by_sensitive_feature:
            roc_convex_hull = _tradeoff_curve(group, sensitive_feature_value, flip=self.flip)
            self._tradeoff_curve[sensitive_feature_value] = \
                _interpolate_curve(roc_convex_hull, 'x', 'y', 'operation', self._x_grid)
            y_values[sensitive_feature_value] = self._tradeoff_curve[sensitive_feature_value]['y']

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
        # Calculate the confusion matrix counts based on the false positive rate
        # (along the x axis) and the true positive rate (along the y axis).
        counts = _extend_confusion_matrix(
            false_positives=(n_negative * self._x_grid),
            true_negatives=(n_negative * (1.0 - self._x_grid)),
            true_positives=(n_positive * self._y_min),
            false_negatives=(n_positive * (1.0 - self._y_min))
        )
        objective_values = np.around(METRIC_DICT[self.objective](counts), 15)
        # Calculate the error at any given x as the sum of
        # a) the proportion of negative labels multiplied by x which represents
        #    the conditional probability P[Y_hat=1 | Y=0], i.e. the probability
        #    of a positive prediction given a negative label.
        # b) the propotion of positive labels multiplied by 1-y_min, where y_min
        #    represents the conditional probability P[Y_hat=1 | Y=1], i.e. the
        #    probability of a correct prediction of a positive label, so 1-y_min
        #    represents a negative prediction given a positive label.
        i_best_EO = objective_values.idxmax()

        self._x_best = self._x_grid[i_best_EO]
        self._y_best = self._y_min[i_best_EO]

        # create the solution as interpolation of multiple points with a separate
        # interpolation per sensitive feature
        interpolation_dict = {}
        for sensitive_feature_value in self._tradeoff_curve.keys():
            roc_result = self._tradeoff_curve[sensitive_feature_value].transpose()[i_best_EO]
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

            interpolation_dict[sensitive_feature_value] = \
                Bunch(p_ignore=p_ignore,
                      prediction_constant=self._x_best,
                      p0=roc_result.p0,
                      operation0=roc_result.operation0,
                      p1=roc_result.p1,
                      operation1=roc_result.operation1)

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("From ROC curves")
        logger.debug("Best point (EO): %s=%.3f, FP rate=%.3f, TP rate=%.3f",
                     self.objective, objective_values[i_best_EO], self._x_best, self._y_best)
        logger.debug(OUTPUT_SEPARATOR)

        return InterpolatedThresholder(
            self.estimator_, interpolation_dict, prefit=True).fit(None, None)


def _reformat_and_group_data(sensitive_features, labels, scores, sensitive_feature_names=None):
    """Reformats the data into a new pandas.DataFrame and group by sensitive feature values.

    The data are provided as three arguments (`sensitive_features`, `labels`, `scores`) and
    the new  DataFrame is grouped by sensitive feature values so that subsequently each group
    can be handled separately.

    Parameters
    ----------
    sensitive_features : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Sensitive features to identify groups by
    labels : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        the labels of the dataset
    scores : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        the scores produced by a predictor's prediction
    sensitive_feature_names : list of strings
        list of names for the sensitive features in case they were not
        implicitly provided (e.g. if `sensitive_features` is of type
        pandas.DataFrame); default None

    Returns
    -------
    pandas.DataFrameGroupBy
        the training data for the mitigator, grouped by sensitive feature value
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

    Parameters
    ----------
    key : str
        the key in `data_dict` at which `additional_data` should be stored;
        `key` should describe the purpose of `additional_data` in `data_dict`
    data_dict : dict
        the dictionary containing all the relevant data; `additional_data`
        will be inserted at the key `key`.
    additional_data : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
        the data to be added to `data_dict` at the specified `key`

    Returns
    -------
    dict
        The updated `data_dict` with reformatted data at the `key` slot
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
