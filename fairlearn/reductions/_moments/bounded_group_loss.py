# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from .moment import LossMoment
from .moment import _GROUP_ID, _LABEL, _LOSS, _PREDICTION, _ALL
from fairlearn._input_validation import _KW_SENSITIVE_FEATURES


class ConditionalLossMoment(LossMoment):
    r"""A moment for constraining the mean loss or the worst-case loss by a group.

    Parameters
    ----------
    loss : {SquareLoss, AbsoluteLoss}
        A loss object with an `eval` method, e.g. `SquareLoss` or
        `AbsoluteLoss`.
    upper_bound : float
        An upper bound on the loss, also referred to as :math:`\\zeta`;
        `upper_bound` is an optional argument that is not always
        required; default None
    no_groups : bool
        indicates whether to calculate the mean loss or group-level losses,
        default False, i.e., group-level losses are the default behavior
    """

    def __init__(self, loss, *, upper_bound=None, no_groups=False):
        super().__init__(loss)
        self.upper_bound = upper_bound
        self.no_groups = no_groups

    def default_objective(self):
        """Return a default objective."""
        return MeanLoss(self.reduction_loss)

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
        self.neg_basis_present = pd.Series(dtype='float64')
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

    def bound(self):
        """Return the vector of bounds.

        Returns
        -------
        pandas.Series
            A vector of bounds on group-level losses
        """
        if self.upper_bound is None:
            raise ValueError("No Upper Bound")
        return pd.Series(self.upper_bound, index=self.index)

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        if lambda_vec is None:
            adjust = pd.Series(1.0, index=self.index)
        else:
            adjust = lambda_vec / self.prob_attr
        return self.tags.apply(lambda row: adjust[row[_GROUP_ID]], axis=1)


# Ensure that ConditionalLossMoment shows up in correct place in documentation
# when it is used as a base class
ConditionalLossMoment.__module__ = "fairlearn.reductions"


class MeanLoss(ConditionalLossMoment):
    """Moment for evaluating the mean loss."""

    def __init__(self, loss):
        super().__init__(loss, upper_bound=None, no_groups=True)


class BoundedGroupLoss(ConditionalLossMoment):
    """Moment for constraining the worst-case loss by a group.

    For more information refer to the :ref:`user guide <bounded_group_loss>`.
    """

    def __init__(self, loss, *, upper_bound=None):
        super().__init__(loss, upper_bound=upper_bound, no_groups=False)