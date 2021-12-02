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
    r"""Create my own description here.

    Parameters
    //Todo

    References
    ----------
    //Todo

    """

    def __init__(self, estimator, threshold_dict, prefit=False,
                 predict_method='deprecated'):
        self.estimator = estimator
        self.threshold_dict = threshold_dict
        self.prefit = prefit
        self.predict_method = predict_method

        self.validate_threshold_dict_keys()

    def validate_threshold_dict_keys(self):
        """Add info"""

        keys = list(self.threshold_dict.keys())

        # check if all keys are of the same dtype
        if len(keys) > 1:
            same_type = all(isinstance(key, type(keys[0])) for key in keys[1:])

            if not same_type:
                warn("The keys of threshold_dict are not of the same type.")

    def reformat_threshold_dict_keys(self):
        """Add info"""

        # If there are multiple sensitive features this reformatting is
        # necessary in order to be able to warn the user if there are sensitive features
        # provided at predict time which are not mentioned in threshold_dict
        new_keys_dict = {}
        for sf_tuple, threshold in self.threshold_dict.items():

            # E.g. from ('A','B') -> 'A,B'
            sf_combined = ''
            for single_sf in sf_tuple:
                sf_combined += '{},'.format(single_sf)
            sf_combined = sf_combined[:-1]

            new_keys_dict[sf_combined] = threshold

        self.threshold_dict = new_keys_dict

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

    def predict(self, X, *, sensitive_features):
        """Predict stuff, write better explanation later."""
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
        sf_values = list(self.threshold_dict.keys())
        if not all(sf_value in sf_values for sf_value in sensitive_feature_vector):
            # Throw warning that we there as combi not in threshold_dict
            # I will improve this warning to also mention which specific values are
            # found in sensitive_feature_vector but not in threshold_dict
            warn('combi not found in threshold_dict')

        final_predictions = 0.0*base_predictions_vector

        for sf, threshold in self.threshold_dict.items():

            operation = ThresholdOperation('>', threshold)

            thresholded_predictions = 1.0 * operation(base_predictions_vector)

            final_predictions[sensitive_feature_vector == sf] = \
                thresholded_predictions[sensitive_feature_vector == sf]

        return final_predictions
