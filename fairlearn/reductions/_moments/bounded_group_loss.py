# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from .moment import LossMoment
from .moment import _GROUP_ID, _LABEL, _LOSS, _PREDICTION, _ALL
from fairlearn._input_validation import _KW_SENSITIVE_FEATURES


class ConditionalLossMoment(LossMoment):
    """A moment that quantifies a loss by group."""

    def __init__(self, loss, no_groups=False):
        super().__init__(loss)
        self.no_groups = no_groups

    def default_objective(self):
        """Return a default objective."""
        return AverageLossMoment(self.reduction_loss)

    def load_data(self, X, y, **kwargs):
        """Load data into the moment object."""
        kwargs_mod = kwargs.copy()
        if self.no_groups:
            kwargs_mod[_KW_SENSITIVE_FEATURES] = pd.Series(y).apply(lambda y: _ALL)
        super().load_data(X, y, **kwargs_mod)
        self.prob_attr = self.tags.groupby(_GROUP_ID).size() / self.total_samples
        self.index = self.prob_attr.index
        self.default_objective_lambda_vec = self.prob_attr

        # fill in the information about the basis
        attr_vals = self.tags[_GROUP_ID].unique()
        self.pos_basis = pd.DataFrame()
        self.neg_basis = pd.DataFrame()
        self.neg_basis_present = pd.Series()
        zero_vec = pd.Series(0.0, self.index)
        i = 0
        for attr in attr_vals:
            self.pos_basis[i] = 0 + zero_vec
            self.neg_basis[i] = 0 + zero_vec
            self.pos_basis[i][attr] = 1
            self.neg_basis_present.at[i] = False
            i += 1

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""
        self.tags[_PREDICTION] = predictor(self.X)
        self.tags[_LOSS] = self.reduction_loss.eval(self.tags[_LABEL], self.tags[_PREDICTION])
        expect_attr = self.tags.groupby(_GROUP_ID).mean()
        self._gamma_descr = str(expect_attr[[_LOSS]])
        return expect_attr[_LOSS]

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec):
        """Return the signed weights."""
        adjust = lambda_vec / self.prob_attr
        signed_weights = self.tags.apply(
            lambda row: adjust[row[_GROUP_ID]], axis=1
        )
        return signed_weights


# Ensure that ConditionalLossMoment shows up in correct place in documentation
# when it is used as a base class
ConditionalLossMoment.__module__ = "fairlearn.reductions"


class AverageLossMoment(ConditionalLossMoment):
    """Moment for Average Loss."""

    def __init__(self, loss):
        super().__init__(loss, no_groups=True)


class GroupLossMoment(ConditionalLossMoment):
    """Moment for Group Loss."""

    def __init__(self, loss):
        super().__init__(loss, no_groups=False)


class SquareLoss:
    """Class to evaluate the square loss."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = (max_val-min_val) ** 2

    def eval(self, y_true, y_pred):  # noqa: A003
        """Evaluate the square loss for the given set of true and predicted values."""
        return (np.clip(y_true, self.min_val, self.max_val)
                - np.clip(y_pred, self.min_val, self.max_val)) ** 2


class AbsoluteLoss:
    """Class to evaluate absolute loss."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = np.abs(max_val-min_val)

    def eval(self, y_true, y_pred):  # noqa: A003
        """Evaluate the absolute loss for the given set of true and predicted values."""
        return np.abs(np.clip(y_true, self.min_val, self.max_val)
                      - np.clip(y_pred, self.min_val, self.max_val))


# Ensure that AbsoluteLoss shows up in correct place in documentation
# when it is used as a base class
AbsoluteLoss.__module__ = "fairlearn.reductions"


class ZeroOneLoss(AbsoluteLoss):
    """Class to evaluate a zero-one loss."""

    def __init__(self):
        super().__init__(0, 1)
