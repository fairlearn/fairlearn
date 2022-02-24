# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from warnings import warn

from ._threshold_operation import ThresholdOperation

from ..utils._common import _get_soft_predictions
from ..utils._input_validation import _validate_and_reformat_input
from ._constants import (
    BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
    BASE_ESTIMATOR_NOT_FITTED_WARNING)


class RejectOptionClassifier(BaseEstimator, MetaEstimatorMixin):
    r"""A classifier that produces group specific classifications inside a critical region.

    :code:`RejectOptionClassifier` looks at the certainty by which binary classification
    decisions are made. For decisions made with high certainty (= with a posterior
    probability close to 0 or 1), the result of the classification remains the same.
    For decisions with low certainty (= with a probability to get 1 close to 0.5), the
    result is disregarded and instances from the favorable group receive the undesirable label,
    whilst instances from the deprived group receive the desirable label. The exact range of
    decisions considered to have low certainty is specified by the critical region.

    Parameters
    ----------
    estimator : object
        A `scikit-learn compatible estimator
        <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
        whose output is postprocessed.

    theta : float
        Number between 0 and 0.5 OR 0.5 and 1 to indicate the critical
        region: [0.5 - theta, 0.5 + theta]
        OR :math:`\max\{p(Y=1|X),1 - p(Y=1|X)\} \leq` theta

    deprived_group : any
        Sensitive feature value to identify the deprived group

    favored_group : any
        Sensitive feature value to identify the favored group

    desired_label : {0, 1}, default=1
        Number representing the desired label

    prefit : bool, default=False
        If True, avoid refitting the given estimator. Note that when used with
        :func:`sklearn.model_selection.cross_val_score`,
        :class:`sklearn.model_selection.GridSearchCV`, this will result in an
        error. In that case, please use ``prefit=False``.

    Notes
    -----
    The procedure is based on the algorithm of
    `Kamiran et al. (2012) <https://ieeexplore.ieee.org/document/6413831>`_ [1]_.

    References
    ----------
    .. [1] F. Kamiran, A. Karim and X. Zhang,
        "Decision Theory for Discrimination-Aware Classification,"
        2012 IEEE 12th International Conference on Data Mining, 2012,
        pp. 924-929, doi: 10.1109/ICDM.2012.45.
        Available: https://ieeexplore.ieee.org/document/6413831

    Examples
    --------
    >>> from fairlearn.postprocessing import RejectOptionClassifier
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X_train = pd.DataFrame([[0, 4], [6, 2], [1, 3], [10, 5], [1, 7], [-2, 1]])
    >>> y_train = pd.Series([1, 0, 1, 0, 1, 0])
    >>> sensitive_features_train = pd.DataFrame(
    ...                 [['A'], ['A'], ['A'], ['B'], ['B'], ['B']], columns=['SF1'])
    >>> estimator = RandomForestClassifier(random_state=1)
    >>> estimator.fit(X_train, y_train)
    >>> X_test = pd.DataFrame([[0, 5], [-1, 6], [2, 2], [8, -11]])
    >>> sensitive_features_test = pd.DataFrame(
    ...                 [['A'], ['A'], ['B'], ['B']], columns=['SF1'])
    >>> estimator.predict_proba(X_test)[:, 1]
    [0.77 0.58 0.44 0.12]
    >>> estimator.predict(X_test)
    [1 1 0 0]
    >>> reject_clf = RejectOptionClassifier(estimator=estimator,
    ...                                     theta=0.6,
    ...                                     deprived_group='B',
    ...                                     favored_group='A',
    ...                                     prefit=True)
    >>> reject_clf.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    >>> reject_clf.predict(X_test, sensitive_features=sensitive_features_test)
    0    1.0
    1    0.0
    2    1.0
    3    0.0
    """

    def __init__(self, estimator, theta, deprived_group, favored_group, desired_label=1,
                 prefit=False):

        self.estimator = estimator
        self.theta = theta
        self.deprived_group = deprived_group
        self.favored_group = favored_group
        self.desired_label = desired_label
        self.prefit = prefit

        self._validate_theta()
        self._validate_desired_label()

    def _validate_theta(self):
        """Check if theta is float and between 0.5 and 1, raise error if not.

        OR between 0 and 0.5
        """
        if not isinstance(self.theta, float):
            raise TypeError("theta should be of type 'float', but is of "
                            "type '{}'".format(type(self.theta).__name__))

        # my way
        # if not 0 < self.theta < 0.5:
        #     raise ValueError("theta should be between 0 and 0.5, but is {}".format(self.theta))

        # paper way
        if not 0.5 < self.theta < 1:
            raise ValueError("theta should be between 0.5 and 1, but is {}".format(self.theta))

    def _validate_desired_label(self):
        """Check if desired_label is 0 or 1, raise error if not."""
        if self.desired_label not in (0, 1):
            raise ValueError("desired_label should be 0 or 1, but is {}"
                             .format(self.desired_label))

    def _check_observed_sf(self, observed_sf_values):
        """Check if the observed sf values match the specified deprived/favored group.

        Parameters
        ----------
        observed_sf_values : list
            The sensitive feature values observed in sensitive_features, inputted by the user
            in fit() or predict()
        """
        unmentioned_sf_values = np.setdiff1d(observed_sf_values,
                                             [self.deprived_group, self.favored_group])

        n_unmentioned_sf = len(unmentioned_sf_values)

        if n_unmentioned_sf > 0:

            if n_unmentioned_sf > 1:
                msg = "The observed sensitive feature values '{}'".format(
                    unmentioned_sf_values[0])

                for i in range(1, n_unmentioned_sf):
                    msg += ", '{}'".format(unmentioned_sf_values[i])

                msg += " do not "

            else:
                msg = "The observed sensitive feature value '{}' does not ".format(
                    unmentioned_sf_values[0])

            msg += "correspond to the specified values of the deprived and favored group: " +\
                "'{}' and '{}'.".format(self.deprived_group, self.favored_group)

            raise ValueError(msg)

    def fit(self, X, y, sensitive_features, **kwargs):
        """Fit the estimator.

        If `prefit` is set to `True` then the base estimator is kept as is.
        Otherwise it is fitted from the provided arguments.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix
        y : numpy.ndarray, list, pandas.DataFrame, or pandas.Series
            The label vector
        sensitive_features : numpy.ndarray, list, pandas.DataFrame,\
            or pandas.Series
            sensitive features to identify groups by
        """
        _, _, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X, y=y, sensitive_features=sensitive_features, expect_y=True,
            enforce_binary_labels=False)

        # check if observed sf values match the specified deprived/favored group
        observed_sf_values = sensitive_feature_vector.unique().tolist()
        self._check_observed_sf(observed_sf_values)

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

    def predict(self, X, *, sensitive_features):
        """Predict as explained in the decription of :code:`RejectOptionClassifier`.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix
        sensitive_features : numpy.ndarray, list, pandas.DataFrame,\
            or pandas.Series
            sensitive features to identify groups by

        Returns
        -------
        numpy.ndarray, list, pandas.DataFrame, or pandas.Series
            The prediction in the form of a scalar or vector.
            If `X` represents the data for a single example the result will be
            a scalar. Otherwise the result will be a vector.
        """
        check_is_fitted(self)

        base_predictions = np.array(
            _get_soft_predictions(self.estimator_, X, 'predict_proba'))

        _, base_predictions_vector, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X, y=base_predictions, sensitive_features=sensitive_features, expect_y=True,
            enforce_binary_labels=False)

        # check if observed sf values match the specified deprived/favored group
        observed_sf_values = sensitive_feature_vector.unique().tolist()
        self._check_observed_sf(observed_sf_values)

        final_predictions = 0.0*base_predictions_vector

        # my way
        # for sf_value in observed_sf_values:

        #     if (sf_value == self.deprived_group and self.desired_label == 1) \
        #            or (sf_value == self.favored_group and self.desired_label == 0):
        #         operation = ThresholdOperation('>', 0.5 - self.theta)
        #     else:
        #         operation = ThresholdOperation('>', 0.5 + self.theta)

        #     thresholded_predictions = 1.0 * operation(base_predictions_vector)

        #     final_predictions[sensitive_feature_vector == sf_value] = \
        #         thresholded_predictions[sensitive_feature_vector == sf_value]

        # paper way
        for sf_value in observed_sf_values:

            if (sf_value == self.deprived_group and self.desired_label == 1) \
                    or (sf_value == self.favored_group and self.desired_label == 0):

                operation = ThresholdOperation('>', 1 - self.theta)

            else:
                operation = ThresholdOperation('>', self.theta)

            thresholded_predictions = 1.0 * operation(base_predictions_vector)

            final_predictions[sensitive_feature_vector == sf_value] = \
                thresholded_predictions[sensitive_feature_vector == sf_value]

        return final_predictions
