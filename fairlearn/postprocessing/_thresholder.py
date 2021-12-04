# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from warnings import warn


from postprocessing._threshold_operation import ThresholdOperation

# try to get it to work now
from utils._common import _get_soft_predictions
from utils._input_validation import _validate_and_reformat_input
from postprocessing._constants import (
    BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
    BASE_ESTIMATOR_NOT_FITTED_WARNING)


# how it should be
# from ..utils._common import _get_soft_predictions
# from ..utils._input_validation import _validate_and_reformat_input
# from ._constants import (
#     BASE_ESTIMATOR_NONE_ERROR_MESSAGE,
#     BASE_ESTIMATOR_NOT_FITTED_WARNING)


class Thresholder(BaseEstimator, MetaEstimatorMixin):
    r"""A classifier that uses group-specific thresholds for prediction.

    Parameters
    ----------
    estimator : object
        A `scikit-learn compatible estimator <https://scikit-learn.org/stable/developers/develop.html#estimators>`_  # noqa
        whose output is postprocessed.

    threshold_dict : dict
        Dictionary that maps sensitive feature values to group-specific thresholds

    prefit : bool, default=False
        If True, avoid refitting the given estimator. Note that when used with
        :func:`sklearn.model_selection.cross_val_score`,
        :class:`sklearn.model_selection.GridSearchCV`, this will result in an
        error. In that case, please use ``prefit=False``.

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
        - 'predict': only use if estimator is a regressor. Uses the regression 
        values given by `predict`.

    Notes
    -----
    The procedure is based on the algorithm of
    `Kamiran et al. (2012) <>
    `Hardt et al. (2016) <https://ieeexplore.ieee.org/document/6413831>`_ [1]_.

    References
    ----------
    .. [1] F. Kamiran, A. Karim and X. Zhang, 
        "Decision Theory for Discrimination-Aware Classification," 
        2012 IEEE 12th International Conference on Data Mining, 2012, 
        pp. 924-929, doi: 10.1109/ICDM.2012.45.
        Available: https://ieeexplore.ieee.org/document/6413831
    """

    def __init__(self, estimator, threshold_dict, prefit=False,
                 predict_method='deprecated'):
        self.estimator = estimator
        self.threshold_dict = threshold_dict
        self.prefit = prefit
        self.predict_method = predict_method

        self.validate_threshold_dict_keys()

    def validate_threshold_dict_keys(self):
        """Check if all keys of :code:`threshold_dict` are of the same type."""
        keys = list(self.threshold_dict.keys())

        if len(keys) > 1:
            same_type = \
                all(isinstance(key, type(keys[0])) for key in keys[1:])

            if not same_type:
                warn("Not all the keys of 'threshold_dict' are of the same\
                type. Please make sure that all keys are of the same type")

    def reformat_threshold_dict_keys(self):
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

    def check_for_unseen_sf_values(self, sensitive_feature_vector):
        """Checks if there are sensitive feature value(s) (cominations),
        that are not mentioned in :code: `threshold_dict`.

        Returns
        -------
        Warning message if there are unseen feature value(s) (cominations),
        None if not
        """
        known_sf_values = list(self.threshold_dict.keys())

        new_sf_values = []

        for sf_value in sensitive_feature_vector:
            if (sf_value not in known_sf_values) and \
                    (sf_value not in new_sf_values):

                new_sf_values.append(sf_value)

        if new_sf_values:

            msg = 'The following groups are provided at predict time,\
                    but not mentioned in `threshold_dict`: '

            for new_sf in new_sf_values:
                msg += ' {}'.format(new_sf)

        return msg if new_sf_values else None

    def fit(self, X, y, **kwargs):
        """Fit the estimator.

        If `prefit` is set to `True` then the base estimator is kept as is.
        Otherwise it is fitted from the provided arguments.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            The label vector
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

    def predict(self, X, *, sensitive_features):
        """Predict using the group-specific thresholds provided in :code: `threshold_dict`.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix
        sensitive_features : numpy.ndarray, list, pandas.DataFrame,\
            or pandas.Series
            sensitive features to identify groups by

        Returns
        -------
        numpy.ndarray
            The prediction in the form of a scalar or vector.
            If `X` represents the data for a single example the result will be
            a scalar. Otherwise the result will be a vector.
        """
        check_is_fitted(self)

        # get soft predictions
        base_predictions = np.array(
            _get_soft_predictions(self.estimator_, X, self._predict_method)
        )

        # validate and reformat input
        _, base_predictions_vector, sensitive_feature_vector, _ = _validate_and_reformat_input(
            X, y=base_predictions, sensitive_features=sensitive_features, expect_y=True,
            enforce_binary_labels=False)

        # If there are multiple sensituive features, reformat threshold_dict keys
        # in order to check if sensitive_feature_vector contains sensitive feature
        # combinations not provided in threshold_dict
        if len(sensitive_features.shape) > 1 and sensitive_features.shape[1] > 1:
            self.reformat_threshold_dict_keys()

        # warn if there are sensitive features not in threshold dict
        potential_msg = \
            self.check_for_unseen_sf_values(sensitive_feature_vector)
        if potential_msg:
            warn(potential_msg)

        final_predictions = 0.0*base_predictions_vector

        for sf, threshold in self.threshold_dict.items():

            operation = ThresholdOperation('>', threshold)

            thresholded_predictions = 1.0 * operation(base_predictions_vector)

            final_predictions[sensitive_feature_vector == sf] = \
                thresholded_predictions[sensitive_feature_vector == sf]

        return final_predictions
