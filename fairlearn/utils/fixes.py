# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import sklearn
from sklearn.externals._packaging.version import parse as parse_version

sklearn_version = parse_version(sklearn.__version__)


def logistic_loss(fit_intercept):
    if sklearn_version >= parse_version("1.1"):
        from sklearn.linear_model._linear_loss import LinearModelLoss
        from sklearn._loss.loss import HalfBinomialLoss

        loss = LinearModelLoss(
            base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept
        )
        return loss.loss, "new"
    else:
        # We can import logistic loss in the old way
        from sklearn.linear_model._logistic import (
            _logistic_loss,
        )

        return _logistic_loss, "old"
