# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from .moment import ClassificationMoment
from .moment import _ALL, _LABEL

from fairlearn._input_validation import _validate_and_reformat_input


_MESSAGE_BAD_COSTS = "costs needs to be a dictionary with keys " \
    "'fp' and 'fn' and non-negative values"


class ErrorRate(ClassificationMoment):
    """Misclassification error."""

    short_name = "Err"

    def __init__(self, *, costs=None):
        """Initialize the costs."""
        super(ErrorRate, self).__init__()
        if costs is None:
            self.fp_cost = 1.0
            self.fn_cost = 1.0
        elif (type(costs) is dict
              and costs.keys() == {'fp', 'fn'}
              and costs['fp'] >= 0.0
              and costs['fn'] >= 0.0):
            self.fp_cost = costs['fp']
            self.fn_cost = costs['fn']
        else:
            raise ValueError(_MESSAGE_BAD_COSTS)
        self.sum_costs = self.fp_cost + self.fn_cost

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        """Load the specified data into the object."""
        _, y_train, sf_train, _ = \
            _validate_and_reformat_input(X, y,
                                         enforce_binary_labels=True,
                                         sensitive_features=sensitive_features,
                                         control_features=control_features)
        # The following uses X  so that the estimators get X untouched
        super().load_data(X, y_train, sensitive_features=sf_train)
        self.index = [_ALL]

    def gamma(self, predictor):
        """Return the gamma values for the given predictor."""
        pred = predictor(self.X)
        if isinstance(pred, np.ndarray):
            # TensorFlow is returning an (n,1) array, which results
            # in the subtraction in the 'error =' line generating an
            # (n,n) array
            pred = np.squeeze(pred)
        signed_errors = self.tags[_LABEL] - pred
        total_fn_cost = np.sum(signed_errors[signed_errors > 0] * self.fn_cost)
        total_fp_cost = np.sum(-signed_errors[signed_errors < 0] * self.fp_cost)
        error_value = (total_fn_cost + total_fp_cost) / self.total_samples
        error = pd.Series(data=error_value,
                          index=self.index)
        self._gamma_descr = str(error)
        return error

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        weights = self.sum_costs * self.tags[_LABEL] - self.fp_cost
        if lambda_vec is None:
            return weights
        else:
            return lambda_vec[_ALL] * weights
