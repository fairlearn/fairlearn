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


class Thresholder(BaseEstimator, MetaEstimatorMixin):
    r"""A classifier that uses group-specific thresholds for prediction.

    Can be used to output binary predictions, based on the output of
    both regressors and classifiers.

    Parameters
    ----------
    estimator : object
        A `scikit-learn compatible estimator
        <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
        whose output is postprocessed.

    threshold_dict : dict
        Dictionary that maps sensitive feature values to group-specific thresholds.
        Of the form {`sensitive_feature_value(s)`: `threshold`}, with:

        - `sensitive_feature_value`: a value SF of the sensitive feature to
          specify a subgroup, or a tuple (SF1,SF2,...) if there are multiple
          sensitive features
        - `threshold`: { `float` , ( '>' , `float` ), ( '<' , `float` )}. The threshold specifies
          when to predict 1. Providing `float` has the same effect as ( '>' , `float` ).

    prefit : bool, default=False
        If True, avoid refitting the given estimator. Note that when used with
        :func:`sklearn.model_selection.cross_val_score`,
        :class:`sklearn.model_selection.GridSearchCV`, this will result in an
        error. In that case, please use ``prefit=False``.

    predict_method : {'auto', 'predict_proba', 'decision_function', 'predict'}
             default='auto'

        Defines which method of the ``estimator`` is used to get the values
        to be thresholded

        - 'auto': use one of `predict_proba`, `decision_function`, or
          `predict`, in that order.
        - 'predict_proba': use the second column from the output of
          `predict_proba`. It is assumed that the second column represents the
          positive outcome.
        - 'decision_function': use the raw values given by the
          `decision_function`.
        - 'predict': only use if estimator is a regressor. Uses the regression
          values given by `predict`.

    default_threshold : {'auto', `float` , ( '>' , `float` ), ( '<' , `float` )}
            default='auto'.

        The default threshold is the threshold to which groups that are not
        mentioned in the ``threshold_dict`` are compared.

        If 'auto':

            - set ``default_threshold`` to 0.5 if ``predict_method`` is either
              `predict_proba` or `predict`
            - set ``default_threshold`` to 0 if ``predict_method`` is `decision_function`

        Providing `float` has the same effect as ( '>' , `float` ).

    Examples
    --------
    The following example shows how to threshold predictions for both data with
    a single sensitive feature and data with multiple sensitive features.

    >>> from fairlearn.postprocessing import Thresholder
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> # Example with one sensitive feature
    >>> X_train = pd.DataFrame([[0, 4], [6, 2], [1, 3], [10, 5], [1, 7], [-2, 1]])
    >>> y_train = pd.Series([1, 1, 1, 0, 0, 0])
    >>> sensitive_features_train = pd.DataFrame(
    ...                   [['A'], ['B'], ['C'], ['A'], ['B'], ['C']], columns=['SF1'])
    >>> estimator = RandomForestClassifier(random_state=1)
    >>> estimator.fit(X_train, y_train)  # doctest: +SKIP
    >>> X_test = pd.DataFrame([[-1, 6], [2, 2], [8, -11]])
    >>> sensitive_features_test = pd.DataFrame([['A'], ['B'], ['C']], columns=['SF1'])
    >>> estimator.predict_proba(X_test)[:, 1]
    array([0.24, 0.82, 0.61])
    >>> estimator.predict(X_test)
    array([0, 1, 1])
    >>> threshold_dict = {'A': .2, 'B': ('<', .6), 'C': ('>', .7)}
    >>> thresholder = Thresholder(estimator=estimator,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')
    >>> thresholder.fit(X_train, y_train, sensitive_features=sensitive_features_train)  # doctest: +SKIP # noqa: E501
    >>> thresholder.predict(X_test, sensitive_features=sensitive_features_test)
    0    1.0
    1    0.0
    2    0.0
    dtype: float64

    >>> # Example with multiple sensitive features
    >>> sensitive_features_train = pd.DataFrame([['A', 'D'],
    ...                                          ['B', 'E'],
    ...                                          ['B', 'D'],
    ...                                          ['A', 'D'],
    ...                                          ['B', 'E'],
    ...                                          ['B', 'D']], columns=['SF1', 'SF2'])
    >>> sensitive_features_test = pd.DataFrame([['A', 'D'],
    ...                                         ['B', 'E'],
    ...                                         ['B', 'D']], columns=['SF1', 'SF2'])
    >>> threshold_dict = {('A', 'D'): .2, ('B', 'E'): ('<', .6), ('B', 'D'): ('>', .7)}
    >>> thresholder = Thresholder(estimator=estimator,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')
    >>> thresholder.fit(X_train, y_train, sensitive_features=sensitive_features_train)  # doctest: +SKIP # noqa: E501
    >>> thresholder.predict(X_test, sensitive_features=sensitive_features_test)
    0    1.0
    1    0.0
    2    0.0
    dtype: float64
    """

    def __init__(self, *, estimator, threshold_dict, prefit=False,
                 predict_method='auto', default_threshold='auto'):
        self.estimator = estimator
        self.threshold_dict = threshold_dict
        self.prefit = prefit

        self._validate_threshold_dict()
        self._validate_and_set_predict_method(predict_method)
        self._validate_and_set_default_threshold(default_threshold)

    def _validate_single_threshold(self, threshold):
        """Check if a threshold is specified correctly.

        Correct means either as `float` or `tuple`. If a threshold is specified in a tuple,
        it should be in either of these two formast: ('>',float) or ('<',float).

        Raise an error if the threshold is specified incorrectly.

        Parameters
        ----------
        threshold: any
            A user-specified threshold
        """
        # Check if thresholds is of type 'float' or 'tuple'
        if not isinstance(threshold, (float, tuple)):
            raise TypeError("All specified thresholds should be of type 'float' or 'tuple', " +
                            "but {} is of type '{}'".format(
                                threshold, type(threshold).__name__))

        # If provided in tuple, check if done so correctly
        if isinstance(threshold, tuple):
            msg = ''
            if threshold[0] not in ['>', '<']:
                msg += "The operator of a specified threshold operation should " +\
                    "be either '>' or '<'. However, for {} it is {}.".format(
                        threshold, threshold[0])
            if not isinstance(threshold[1], float):
                if msg:
                    msg += " The threshold should be of type 'float', " +\
                        "however {} is of type '{}'.".format(
                            threshold[1], type(threshold[1]).__name__)
                else:
                    msg += "The threshold of a specified threshold " +\
                        "operation should be of type 'float'. " +\
                        "However, for {} it is of type '{}'.".format(
                            threshold, type(threshold[1]).__name__)
            if msg:
                raise ValueError(msg)

    def _validate_threshold_dict(self):
        """Check if :code: `threshold_dict` is specified correctly.

        For the keys (subgroups/sensitive feature values), check if all keys are of the same type.
        Warn the user if this is not the case.

        For the values (thresholds), check if they are provided correctly by calling
        self._validate_single_threshold().
        """
        keys, values = zip(*self.threshold_dict.items())

        # Check if all keys are of the same type
        if len(keys) > 1:
            for key in keys[1:]:
                if not isinstance(key, type(keys[0])):
                    warn("Not all the keys of 'threshold_dict' are of the same type. "
                         "{} is of type '{}', while {} is of type '{}'. "
                         "Please make sure that all keys are of the same type.".format(
                             keys[0], type(keys[0]).__name__, key, type(key).__name__))
                    break

        # Check the provided thresholds
        for threshold in values:
            self._validate_single_threshold(threshold)

    def _validate_and_set_predict_method(self, predict_method):
        """Validate and set predict_method.

        If 'predict_method' is set to 'auto', set self.predict_method according
        to predetermined logic.
        If 'predict_method' is not set to 'auto', check correctness and set
        self.predict_method if it is deemed correct, otherwise raise a ValueError.

        Parameters
        ----------
        predict_method : any
            The specified predict_method
        """
        if predict_method == "auto":
            if hasattr(self.estimator, "predict_proba"):
                self.predict_method = "predict_proba"
            elif hasattr(self.estimator, "decision_function"):
                self.predict_method = "decision_function"
            else:
                self.predict_method = "predict"
        # If not 'auto', check correctness (=should be one of there three)
        elif predict_method in ("predict_proba", "decision_function", "predict"):
            self.predict_method = predict_method
        # Raise error if predict_method is not one of
        # ("auto", "predict_proba", "decision_function", "predict")
        else:
            raise ValueError("'predict_method' should be any of 'auto'" +
                             ", 'predict_proba', 'decision_function', 'predict', but is '{}'"
                             .format(predict_method))

    def _validate_and_set_default_threshold(self, default_threshold):
        """Validate and set the default_threshold.

        If 'default_theshold' is set to 'auto', set self.default_theshold according
        to predetermined logic.
        If 'default_theshold' is not set to 'auto', check correctness and set
        self.default_theshold if it is deemed correct, otherwise raise an Error.

        Parameters
        ----------
        default_threshold : any
            The specified default_threshold
        """
        if default_threshold == "auto":
            if self.predict_method == "predict_proba" or self.predict_method == "predict":
                self.default_threshold = 0.5

            # because ensuring that predict_method is either one of the three, this
            # should only happen if it is decision_function
            else:
                self.default_threshold = 0.0

        else:
            self._validate_single_threshold(default_threshold)
            # if we get here, no error was raised so we can safely set self.default_threshold
            self.default_threshold = default_threshold

    def _reformat_threshold_dict_keys(self):
        """Reformats the keys of the provided :code: `threshold_dict`.

        This is necessary to check to which group a sample belongs, after
        :code: `_validate_and_reformat_input` is called on the data.
        """
        new_keys_dict = {}
        for sf_tuple, threshold in self.threshold_dict.items():

            # E.g. from ('A','B') -> 'A,B'
            sf_combined = ''
            for single_sf in sf_tuple:
                sf_combined += '{},'.format(single_sf)
            sf_combined = sf_combined[:-1]

            new_keys_dict[sf_combined] = threshold

        self.threshold_dict = new_keys_dict

    def _check_for_unseen_sf_values(self, sensitive_feature_vector):
        """Check for unseen sensitive feature values.

        Checks if there are sensitive feature value(s) (combinations),
        that are not mentioned in :code: `threshold_dict`.

        Returns
        -------
        Error message if there are unseen feature value(s) (combinations),
        None if not
        """
        new_sf_values = []

        for sf_value in sensitive_feature_vector:
            if (sf_value not in self.known_sf_values) and \
                    (sf_value not in new_sf_values):

                new_sf_values.append(sf_value)

        if new_sf_values:

            msg = "The following groups are provided at predict time, " +\
                  "but were not observed at fit time:"

            for new_sf in new_sf_values:
                msg += ' {}'.format(new_sf)

        return msg if new_sf_values else None

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
        # get list of groups seen at training time, to compare with groups
        # at predict time
        _, _, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X, y=y, sensitive_features=sensitive_features, expect_y=True,
            enforce_binary_labels=False)

        self.known_sf_values = sensitive_feature_vector.unique()

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
        """Predict using the group-specific thresholds provided in ``threshold_dict``.

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

        # get soft predictions
        base_predictions = np.array(
            _get_soft_predictions(self.estimator_, X, self.predict_method)
        )

        # validate and reformat input
        _, base_predictions_vector, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X, y=base_predictions, sensitive_features=sensitive_features, expect_y=True,
            enforce_binary_labels=False)

        # If there are multiple sensitive features, reformat threshold_dict keys
        # in order to check if sensitive_feature_vector contains sensitive feature
        # combinations not provided in threshold_dict
        if len(sensitive_features.shape) > 1 and sensitive_features.shape[1] > 1:
            self._reformat_threshold_dict_keys()

        # Raise error if there are sensitive features not in threshold dict
        potential_msg = self._check_for_unseen_sf_values(sensitive_feature_vector)
        if potential_msg:
            raise ValueError(potential_msg)

        final_predictions = 0.0*base_predictions_vector

        def _make_predictions(sf, threshold):
            """Predict all instances of a specific subgroup.

            Parameters
            ----------
            sf : any
                value of sensitive feature to identify subgroup by
            threshold : { `float` , ( '>' , `float` ), ( '<' , `float` )}
                threshold to base the prediction on
            """
            if isinstance(threshold, float):
                operation = ThresholdOperation('>', threshold)
            else:
                operation = ThresholdOperation(threshold[0], threshold[1])

            thresholded_predictions = 1.0 * operation(base_predictions_vector)

            final_predictions[sensitive_feature_vector == sf] = \
                thresholded_predictions[sensitive_feature_vector == sf]

        # predict for the groups mentioned in threshold_dict
        for sf, threshold in self.threshold_dict.items():
            _make_predictions(sf, threshold)

        # Predict for groups not mentioned in threshold_dict
        for unmentioned_sf in [sf for sf in self.known_sf_values if
                               sf not in self.threshold_dict.keys()]:
            _make_predictions(unmentioned_sf, self.default_threshold)

        return final_predictions
