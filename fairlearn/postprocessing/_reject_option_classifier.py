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

    :code:`RejectOptionClassifier` considers the certainty of binary classification predictions
    decisions are made. Predictions made with high certainty (i.e., with a probability close to
    0 or 1), the result of the classification remains the same.
    For decisions with low certainty (i.e., with a probability close to 0.5),
    instances from the :code:`group_to_upselect` receive the
    :code:`selection_label`, whilst instances from the :code:`group_to_downselect` receive
    the opposite label. The critical region defines the range of probabilities in which
    predictions are considered to be of low certainty.

    Parameters
    ----------
    estimator : object
        A `scikit-learn compatible estimator
        <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
        whose output is postprocessed.

    critical_width : float
        Number between 0 and 1 to indicate the critical
        region: [0.5 - critical_width/2, 0.5 + critical_width/2]

    group_to_upselect : any
        Sensitive feature value to identify the group that receives the
        :code:`selection_label` inside the critical region

    group_to_downselect : any
        Sensitive feature value to identify the group that does not receive
        the :code:`selection_label` inside the critical region

    selection_label : {0, 1}, default=1
        Number representing the label received by the :code:`group_to_upselect`
        inside the critical region

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
    >>> estimator.fit(X_train, y_train)  # doctest: +SKIP
    >>> X_test = pd.DataFrame([[0, 5], [-1, 6], [2, 2], [8, -11]])
    >>> sensitive_features_test = ['A', 'A', 'B', 'B']
    >>> estimator.predict_proba(X_test)[:, 1]
    array([0.77, 0.58, 0.44, 0.12])
    >>> estimator.predict(X_test)
    array([1, 1, 0, 0])
    >>> reject_clf = RejectOptionClassifier(estimator=estimator,
    ...                                     critical_width=0.2,
    ...                                     group_to_upselect='B',
    ...                                     group_to_downselect='A',
    ...                                     prefit=True)
    >>> reject_clf.fit(X_train, y_train, sensitive_features=sensitive_features_train)  # doctest: +SKIP # noqa: E501
    >>> reject_clf.predict(X_test, sensitive_features=sensitive_features_test)
    0    1.0
    1    0.0
    2    1.0
    3    0.0
    dtype: float64
    """

    def __init__(self, *, estimator, critical_width, group_to_upselect,
                 group_to_downselect, selection_label=1, prefit=False):

        self.estimator = estimator
        self.critical_width = critical_width
        self.group_to_upselect = group_to_upselect
        self.group_to_downselect = group_to_downselect
        self.selection_label = selection_label
        self.prefit = prefit

        self._validate_critical_width()
        self._validate_selection_label()

    def _validate_critical_width(self):
        """Check if critical_width is float between 0 and 1 (inclusive), raise error if not."""
        # check if float
        if not isinstance(self.critical_width, float):
            raise TypeError("critical_width should be of type 'float', but is of "
                            "type '{}'".format(type(self.critical_width).__name__))

        # check if between 0 and 1
        if not 0 <= self.critical_width <= 1:
            raise ValueError(
                "critical_width should be between 0 and 1, but is {}".format(self.critical_width))

    def _validate_selection_label(self):
        """Check if selection_label is 0 or 1, raise error if not."""
        if self.selection_label not in (0, 1):
            raise ValueError("selection_label should be 0 or 1 (of type 'int'), but is {}"
                             .format(self.selection_label))

    def _check_observed_sf(self, observed_sf_values):
        """Check if the observed sf values match the specified group to (up/down)select.

        Parameters
        ----------
        observed_sf_values : list
            The sensitive feature values observed in sensitive_features, inputted by the user
            in fit() or predict()
        """
        unmentioned_sf_values = np.setdiff1d(observed_sf_values,
                                             [self.group_to_upselect, self.group_to_downselect])

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

            msg += "correspond to the specified values of the group_to_upselect and " +\
                "group_to_downselect: '{}' and '{}'.".format(
                    self.group_to_upselect, self.group_to_downselect)

            raise ValueError(msg)

    def fit(self, X, y, *, sensitive_features, **kwargs):
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

        # make final predictions
        for sf_value in observed_sf_values:

            if (sf_value == self.group_to_upselect and self.selection_label == 1) \
                    or (sf_value == self.group_to_downselect and self.selection_label == 0):
                operation = ThresholdOperation('>', 0.5 - self.critical_width/2)
            else:
                operation = ThresholdOperation('>', 0.5 + self.critical_width/2)

            thresholded_predictions = 1.0 * operation(base_predictions_vector)

            final_predictions[sensitive_feature_vector == sf_value] = \
                thresholded_predictions[sensitive_feature_vector == sf_value]

        return final_predictions
