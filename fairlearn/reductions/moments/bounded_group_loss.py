# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from .moment import Moment
from .moment import _REDUCTION_TYPE_LOSS_MINIMIZATION, _GROUP_ID, _LABEL, _LOSS, _PREDICTION, _ALL


class LossMoment(Moment):
    """A moment that quantifies a loss by group"""

    def __init__(self, loss, no_groups=False):
        super().__init__()
        self.reduction_type = _REDUCTION_TYPE_LOSS_MINIMIZATION
        self.reduction_loss = loss
        self.no_groups = no_groups

    def default_objective(self):
        return AverageLossMoment(self.reduction_loss)

    def init(self, dataX, dataA, dataY):
        if self.no_groups:
            dataA = pd.Series(dataY).apply(lambda y: _ALL)
        super().init(dataX, dataA, dataY)
        self.prob_attr = self.tags.groupby(_GROUP_ID).size() / self.n
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
        """ Calculates the degree to which constraints are currently violated by
        the predictor.
        """
        self.tags[_PREDICTION] = predictor(self.X)
        self.tags[_LOSS] = self.reduction_loss.eval(self.tags[_LABEL], self.tags[_PREDICTION])
        expect_attr = self.tags.groupby(_GROUP_ID).mean()
        self._gamma_descr = str(expect_attr[[_LOSS]])
        return expect_attr[_LOSS]

    def project_lambda(self, lambda_vec):
        return lambda_vec

    def signed_weights(self, lambda_vec):
        adjust = lambda_vec / self.prob_attr
        signed_weights = self.tags.apply(
            lambda row: adjust[row[_GROUP_ID]], axis=1
        )
        return signed_weights


class AverageLossMoment(LossMoment):

    def __init__(self, loss):
        super().__init__(loss, no_groups=True)


class GroupLossMoment(LossMoment):

    def __init__(self, loss):
        super().__init__(loss, no_groups=False)


class SquareLoss:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = (max_val-min_val) ** 2

    def eval(self, y_true, y_pred):
        return (np.clip(y_true, self.min_val, self.max_val)
                - np.clip(y_pred, self.min_val, self.max_val)) ** 2


class AbsoluteLoss:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = np.abs(max_val-min_val)

    def eval(self, y_true, y_pred):
        return np.abs(np.clip(y_true, self.min_val, self.max_val)
                      - np.clip(y_pred, self.min_val, self.max_val))


class ZeroOneLoss(AbsoluteLoss):

    def __init__(self):
        super().__init__(0, 1)
