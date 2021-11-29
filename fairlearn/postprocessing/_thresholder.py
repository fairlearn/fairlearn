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

        final_predictions = 0.0*base_predictions_vector

        for sf, threshold in self.threshold_dict.items():

            operation = ThresholdOperation('>', threshold)

            thresholded_predictions = 1.0 * operation(base_predictions_vector)

            final_predictions[sensitive_feature_vector == sf] = \
                thresholded_predictions[sensitive_feature_vector == sf]

        return final_predictions
