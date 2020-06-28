# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random

from warnings import warn

from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from fairlearn._input_validation import _validate_and_reformat_input
from ._constants import (
    BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
    BASE_ESTIMATOR_NOT_FITTED_WARNING)


class InterpolatedThresholder(BaseEstimator, MetaEstimatorMixin):
    """Binary predictor that thresholds continuous predictions of a base estimator.

    At prediction time, the predictor takes as input both standard and sensitive features.
    Based on the values of sensitive features, it then applies a randomized thresholding
    transformation according to the provided `interpolation_dict`.

    :param estimator: base estimator
    :param dict interpolation_dict: maps sensitive feature values to `Bunch` that describes the
        interpolation transformation via the following fields:

        - p0, operation0: with probability p0, operation0 is executed
        - p1, operation1: with probability p1, operation1 is executed
        - p_ignore, prediction_constant: two optional fields; if present then the result of
          the draw of operation0 or operation1 is kept with probability 1 - p_ignore, and gets
          replaced by prediction_constant with probability p_ignore.

        The numbers p0 and p1 must be non-negative and add up to 1, operation0 and
        operation1 must be instances of :class:`ThresholdOperation`, and p_ignore must be
        between 0 and 1.
    :param bool prefit: if `True` then the base estimator is not fitted in :meth:`fit`.
    """

    def __init__(self, estimator, interpolation_dict, prefit=False):
        self.estimator = estimator
        self.interpolation_dict = interpolation_dict
        self.prefit = prefit

    def fit(self, X, y, **kwargs):
        """Fit the estimator.

        If `prefit` is set to `True` then the base estimator is kept as is.
        Otherwise it is fitted from the provided arguments.
        """
        if self.estimator is None:
            raise ValueError(BASE_ESTIMATOR_NONE_ERROR_MESSAGE)

        if not self.prefit:
            self.estimator_ = clone(self.estimator).fit(X, y, **kwargs)
        else:
            try:
                check_is_fitted(self.estimator)
            except NotFittedError:
                warn(BASE_ESTIMATOR_NOT_FITTED_WARNING.format(type(self).__name__))
            self.estimator_ = self.estimator
        return self

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
        base_predictions = np.array(self.estimator_.predict(X))
        _, base_predictions_vector, sensitive_feature_vector = _validate_and_reformat_input(
            X, y=base_predictions, sensitive_features=sensitive_features, expect_y=True,
            enforce_binary_labels=False)

        positive_probs = 0.0*base_predictions_vector
        for a, interpolation in self.interpolation_dict.items():
            interpolated_predictions = \
                interpolation.p0 * interpolation.operation0(base_predictions_vector) + \
                interpolation.p1 * interpolation.operation1(base_predictions_vector)
            if 'p_ignore' in interpolation:
                interpolated_predictions = \
                    interpolation.p_ignore * interpolation.prediction_constant + \
                    (1 - interpolation.p_ignore) * interpolated_predictions
            positive_probs[sensitive_feature_vector == a] = \
                interpolated_predictions[sensitive_feature_vector == a]
        return np.array([1.0 - positive_probs, positive_probs]).transpose()

    def predict(self, X, *, sensitive_features, random_state=None):
        """Provide a prediction for the given input data.

        Note that this is non-deterministic, due to the nature of the
        interpolation.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame

        :return: The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        :rtype: Scalar or vector
        """
        check_is_fitted(self)
        if random_state:
            random.seed(random_state)
        positive_probs = self._pmf_predict(
            X, sensitive_features=sensitive_features)[:, 1]
        return (positive_probs >= np.random.rand(len(positive_probs))) * 1
