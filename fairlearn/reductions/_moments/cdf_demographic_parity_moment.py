# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from .utility_parity import UtilityParity
from .moment import Moment, _LABEL, _GROUP_ID
from .weighted_error_rate import WeightedErrorRate
import pandas as pd
import numpy as np
from itertools import repeat


class CDF_DemographicParity(Moment):
    r"""A moment for constraining the demographic parity for regression tasks.

    Parameters
    ----------
    loss : {SquareLoss, AbsoluteLoss}
        A loss object with an `eval` method, e.g. `SquareLoss` or
        `AbsoluteLoss`.
    y_range : (float,float)
        A tiple that specifies the range of y labels of the data.
    differece_bound : bool
        The constraints' difference bound for constraints that are expressed
        as differences, also referred to as :math:`\\epsilon` in documentation.
        Default None
    grids: list of floats
        Instead of supplying a number for the grid, users may specify the exact
        set of grids they desire using this argument.
    grid_num: int
        If use does not specify grids, then grid_num and y_range is used to calculate the grids
        used as thresholds.
    """
    short_name = "CDF_DemographicParity"

    def __init__(self, loss, y_range, difference_bound=None, grids=[], grid_num=41):
        super().__init__()
        self.utility_parity = UtilityParity(difference_bound=difference_bound)
        self.loss = loss
        self.objective = WeightedErrorRate()
        self.grids = grids
        self.grid_num = grid_num
        self.y_range = y_range

        # check the validity of y_range
        assert (self.y_range is not None), "You should specify the range of y"
        assert self.y_range[0] <= self.y_range[1], "The range of the label y is not valid!"

        # determine the grids
        if self.grids.size == 0:
            self.grids = np.linspace(self.y_range[0], self.y_range[1], self.grid_num)
        else:
            # check whether all grids fall into the empirical range of y
            assert (self.grids <= self.y_range[1]).all()\
                    and (self.grids >= self.y_range[0]).all(),\
                    "Some of the grids do not fall into the range of y"
            self.grids = np.unique(self.grids)
            # ensure grids are listed in an increasing order and are unique

        # in case the grids have already included the two endpoints of the range
        # prepare for the calculation of optimal labels
        self.pred_grid = [self.y_range[0]] +\
            list([self.next_grid(grid) for grid in self.grids]) + [self.y_range[1]]
        self.pred_grid = list(filter(lambda x: x >= self.y_range[0], self.pred_grid))
        self.pred_grid = list(filter(lambda x: x <= self.y_range[1], self.pred_grid))
        self.pred_vec = {}
        for pred in self.pred_grid:
            self.pred_vec[pred] = (1 * (pred >= pd.Series(self.grids)))

    def prev_grid(self, theta):
        index = np.where(self.grids == theta)[0][0]
        if index == 0:
            return self.grids[index] - (self.grids[1]-self.grids[0])/2
        else:
            return (self.grids[index] + self.grids[index-1])/2

    def next_grid(self, theta):
        index = np.where(self.grids == theta)[0][0]
        if index + 1 == self.grids.size:
            return self.grids[index] + (self.grids[1]-self.grids[0])/2
        else:
            return (self.grids[index+1] + self.grids[index])/2

    def augment_data(self, x, a, y):
        self.n = np.shape(x)[0]
        self.width = self.grids[1] - self.grids[0]

        X_aug = pd.concat(repeat(x, self.grid_num))
        A_aug = pd.concat(repeat(a, self.grid_num))
        Y_values = pd.concat(repeat(y, self.grid_num))

        theta_list = [s for theta in self.grids for s in repeat(theta, self.n)]
        X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)

        X_aug.index = range(self.n * self.grid_num)
        A_aug.index = range(self.n * self.grid_num)
        Y_values.index = range(self.n * self.grid_num)

        def weight_assign(theta, y):
            return (self.loss.eval([self.next_grid(_theta) for _theta in theta], y) -
                    self.loss.eval([self.prev_grid(_theta) for _theta in theta], y))
        W = weight_assign(X_aug['theta'], Y_values)
        Y_aug = 1*(W < 0)
        W = abs(W)
        return X_aug, A_aug, Y_aug, W

    def load_data(self, X, y, **kwargs):
        super().load_data(X, y, **kwargs)
        # augment data
        X_aug, A_aug, Y_aug, W = self.augment_data(self.X, self.tags[_GROUP_ID], self.tags[_LABEL])
        self.objective.load_augmented_data(X_aug, Y_aug, sensitive_features=A_aug, weights=W)

        # pass augmented data to utility_parity
        self.utility_parity.load_data(X_aug, Y_aug, event=X_aug["theta"], sensitive_features=A_aug)
        self.index = self.utility_parity.index

        empirical_y_range = (self.tags[_LABEL].min(), self.tags[_LABEL].max())
        assert (self.y_range[0] <= empirical_y_range[0])\
            and (self.y_range[1] >= empirical_y_range[1]),\
            "The actual range of the label y exceeds that required!"

    def default_objective(self):
        return self.objective

    def bound(self):
        return self.utility_parity.bound()

    def project_lambda(self, lambda_vec):
        return self.utility_parity.project_lambda(lambda_vec)

    def gamma(self, predictor):
        def classifier(X):
            x = X.drop(["theta"], axis=1).iloc[:self.n]
            pred = pd.Series(predictor(x))
            pred_binary = pd.concat(repeat(pred, self.grid_num))
            pred_binary.index = range(self.n * self.grid_num)
            return 1*(pred_binary - X['theta'] >= 0)
            # made improvements in constrast to the original line
            # return 1*( predictor(X.drop(['theta'], axis=1)) - X['theta'] >= 0)
            # noting that,after dropping the theta column, the data are repeated for grid_num times
        return self.utility_parity.gamma(classifier)

    def signed_weights(self, lambda_vec):
        return self.utility_parity.signed_weights(lambda_vec)

    def optimal_label(self, signed_weights):
        redY = 1 * (signed_weights > 0)
        redW = signed_weights.abs()
        redW = self.n * self.grid_num * redW / redW.sum()
        minimizer = {}
        for i in range(self.n):
            index_set = [i + j * self.n for j in range(self.grid_num)]
            W_i = redW.iloc[index_set]
            Y_i = redY.iloc[index_set]
            Y_i.index = range(len(Y_i))
            W_i.index = range(len(Y_i))
            cost_i = {}
            for pred in self.pred_grid:
                cost_i[pred] = abs(Y_i - self.pred_vec[pred]).dot(W_i)
            minimizer[i] = min(cost_i, key=cost_i.get)
        return pd.Series(minimizer)
