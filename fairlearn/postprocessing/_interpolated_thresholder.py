# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from warnings import warn

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ..utils._common import _get_soft_predictions
from ..utils._input_validation import _validate_and_reformat_input
from ._constants import (
    BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
    BASE_ESTIMATOR_NOT_FITTED_WARNING,
)


class InterpolatedThresholder(BaseEstimator, MetaEstimatorMixin):
    r"""Binary predictor that thresholds continuous predictions of a base estimator.

    At prediction time, the predictor takes as input both standard and sensitive features.
    Based on the values of sensitive features, it then applies a randomized thresholding
    transformation according to the provided `interpolation_dict`.

    Read more in the :ref:`User Guide <postprocessing>`.

    Parameters
    ----------
    estimator :
        base estimator

    interpolation_dict : dict
        maps sensitive feature values to `Bunch` that describes the
        interpolation transformation via the following fields:

        - p0, operation0: with probability p0, operation0 is executed
        - p1, operation1: with probability p1, operation1 is executed
        - p_ignore, prediction_constant: two optional fields; if present then the result of
          the draw of operation0 or operation1 is kept with probability 1 - p_ignore, and gets
          replaced by prediction_constant with probability p_ignore.

        The numbers p0 and p1 must be non-negative and add up to 1, operation0 and
        operation1 must be instances of :class:`ThresholdOperation`, and p_ignore must be
        between 0 and 1.

    prefit : bool
        if `True` then the base estimator is not fitted in :meth:`fit`.

    predict_method : {'auto', 'predict_proba', 'decision_function', 'predict'\
            }, default='auto'

        Defines which method of the ``estimator`` is used to get the output
        values.

        - 'auto': use one of ``predict_proba``, ``decision_function``, or
          ``predict``, in that order.
        - 'predict_proba': use the second column from the output of
          `predict_proba`. It is assumed that the second column represents the
          positive outcome.
        - 'decision_function': use the raw values given by the
          `decision_function`.
        - 'predict': use the hard values reported by the `predict` method if
          estimator is a classifier, and the regression values if estimator is
          a regressor.
          This is equivalent to what is done in :footcite:`hardt2016equality`.

        .. versionadded:: 0.7
            In previous versions only the ``predict`` method was used
            implicitly.

        .. versionchanged:: 0.7
            From version 0.7, 'predict' is deprecated as the default value and
            the default will change to 'auto' from v0.10.

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self, estimator, interpolation_dict, prefit=False, predict_method="deprecated"
    ):
        self.estimator = estimator
        self.interpolation_dict = interpolation_dict
        self.prefit = prefit
        self.predict_method = predict_method

    def fit(self, X, y, **kwargs):
        """Fit the estimator.

        If `prefit` is set to `True` then the base estimator is kept as is.
        Otherwise it is fitted from the provided arguments.
        """
        if self.estimator is None:
            raise ValueError(BASE_ESTIMATOR_NONE_ERROR_MESSAGE)

        if self.predict_method == "deprecated":
            warn(
                "'predict_method' default value is changed from 'predict' to "
                "'auto'. Explicitly pass `predict_method='predict' to "
                "replicate the old behavior, or pass `predict_method='auto' "
                "or other valid values to silence this warning.",
                FutureWarning,
            )
            self._predict_method = "predict"
        else:
            self._predict_method = self.predict_method

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

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature matrix
        sensitive_features : numpy.ndarray, list, pandas.DataFrame, pandas.Series
            Sensitive features to identify groups by

        Returns
        -------
        numpy.ndarray
            array of tuples with probabilities for predicting 0 or 1, respectively.
            The sum of the two numbers in each tuple needs to add up to 1.
        """
        check_is_fitted(self)
        base_predictions = np.array(
            _get_soft_predictions(self.estimator_, X, self._predict_method)
        )
        (
            _,
            base_predictions_vector,
            sensitive_feature_vector,
            _,
        ) = _validate_and_reformat_input(
            X,
            y=base_predictions,
            sensitive_features=sensitive_features,
            expect_y=True,
            enforce_binary_labels=False,
        )

        positive_probs = 0.0 * base_predictions_vector
        for a, interpolation in self.interpolation_dict.items():
            interpolated_predictions = interpolation.p0 * interpolation.operation0(
                base_predictions_vector
            ) + interpolation.p1 * interpolation.operation1(base_predictions_vector)
            if "p_ignore" in interpolation:
                interpolated_predictions = (
                    interpolation.p_ignore * interpolation.prediction_constant
                    + (1 - interpolation.p_ignore) * interpolated_predictions
                )
            positive_probs[sensitive_feature_vector == a] = interpolated_predictions[
                sensitive_feature_vector == a
            ]
        return np.array([1.0 - positive_probs, positive_probs]).transpose()

    def predict(self, X, *, sensitive_features, random_state=None):
        """Provide a prediction for the given input data.

        Note that this is non-deterministic, due to the nature of the
        interpolation.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        sensitive_features : numpy.ndarray or pandas.DataFrame
            Sensitive features to identify groups by
        random_state : int or :class:`numpy.random.RandomState` instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.

        Returns
        -------
        Scalar or vector as numpy.ndarray
            The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        """
        check_is_fitted(self)
        random_state = check_random_state(random_state)
        positive_probs = self._pmf_predict(X, sensitive_features=sensitive_features)[
            :, 1
        ]
        return (positive_probs >= random_state.rand(len(positive_probs))) * 1
