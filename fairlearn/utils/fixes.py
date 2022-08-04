# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import sklearn
from sklearn.externals._packaging.version import parse as parse_version

sklearn_version = parse_version(sklearn.__version__)


def logistic_loss(fit_intercept):
    """
    Work around updated loss methods in scikit-learn.

    Starting at scikit-learn 1.1, the loss methods have been reworked.
    This function will determine the current scikit-learn version,
    and use the correct implementation of the loss function depending on that.
    See also:
    https://github.com/fairlearn/fairlearn/pull/1043#issuecomment-1149689896
    """
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
