# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from .moment import ClassificationMoment
from .moment import _ALL, _LABEL


class ErrorRate(ClassificationMoment):
    """Misclassification error."""

    short_name = "Err"

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y, **kwargs)
        self.index = [_ALL]

    def gamma(self, predictor):
        """Return the gamma values for the given predictor."""
        pred = predictor(self.X)
        error = pd.Series(data=(self.tags[_LABEL] - pred).abs().mean(),
                          index=self.index)
        self._gamma_descr = str(error)
        return error

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        if lambda_vec is None:
            return 2 * self.tags[_LABEL] - 1
        else:
            return lambda_vec[_ALL] * (2 * self.tags[_LABEL] - 1)
