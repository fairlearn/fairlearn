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
        A tuple that specifies the range of y labels of the data.
    difference_bound : float
        The constraints' difference bound for constraints that are expressed
        as differences, also referred to as :math:`\\epsilon` in documentation.
        Default None
    grids: list of floats
        Instead of supplying a number for the grid, users may specify the exact
        set of grids they desire using this argument. 
        As explained in section 4.1 of 'Agarwal et al. (2019) <https://arxiv.org/abs/1905.12843v1>'
        continuous predictions are discretized to this set of grids.
    grid_num: int
        If the user does not specify `grids`, then `grid_num` and `y_range` is used to calculate the grids
        used as thresholds.
    """
    short_name = "CDF_DemographicParity"

    def __init__(self, loss, y_range, difference_bound=None, grids=None, grid_num=41):
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
        if self.grids == None:
            self.grids = np.linspace(self.y_range[0], self.y_range[1], self.grid_num)
        elif (self.grids <= self.y_range[1]).all() and (self.grids >= self.y_range[0]).all():
            # check whether all grids fall into the empirical range of y
            self.grids = np.unique(self.grids)
            # ensure grids are listed in an increasing order and are unique
        else:
            raise Exception("Some of the grids do not fall into the range of y")
            
        
        # The following variables will be used in the function optimal_label,
        # where a heuristic is used to calculate the optimal y by choosing from 
        # the set of `self.pred_grid` the grid that leads to the lowesr cost
        # This calculation is placed in the __init__ function to avoid unnecessarily 
        # repetious calcutations for the function optimal_label will be called for
        # multiple times.
        self.pred_grid = [self.y_range[0]] +\
            list([self._next_grid(grid) for grid in self.grids]) + [self.y_range[1]]
        # filter grids that are lower than the `self.y_range[0]`
        # this will be necessary in cases the specified grids have already included two endpoints of the range
        self.pred_grid = list(filter(lambda x: x >= self.y_range[0], self.pred_grid))
        # filter grids that are higher than the `self.y_range[1]`
        self.pred_grid = list(filter(lambda x: x <= self.y_range[1], self.pred_grid))
        self.pred_vec = {}
        for pred in self.pred_grid:
            self.pred_vec[pred] = (1 * (pred >= pd.Series(self.grids)))

    def _prev_grid(self, cur_grid):
        index = np.where(self.grids == cur_grid)[0][0]
        if index == 0:
            return self.grids[index] - (self.grids[1]-self.grids[0])/2
        else:
            return (self.grids[index] + self.grids[index-1])/2

    def _next_grid(self, cur_grid):
        index = np.where(self.grids == cur_grid)[0][0]
        if index + 1 == self.grids.size:
            return self.grids[index] + (self.grids[1]-self.grids[0])/2
        else:
            return (self.grids[index+1] + self.grids[index])/2

    def augment_data(self, X, sensitive_features, y):
        """For each x, we create self.grid_num copies and add a new column named 
        "theta" for each copy. The values of this column is the chosen grids. 

        The weight for each new data point is calculated in the section 4.1 of 
        'Agarwal et al. (2019) <https://arxiv.org/abs/1905.12843v1>', but consider
        the more general case where the grids are not equally spaced.

        The binary label for each new data point is determined by whether the 
        corresponding weight is positive or not.


        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        sensitive features: numpy.ndarray or pandas.DataFrame
        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            Label vector

        
        Returns
        -------
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
            the augmented features, sensitive features, labels, and weights.
        """
        self.n = np.shape(X)[0]
        self.width = self.grids[1] - self.grids[0]

        X_aug = pd.concat(repeat(X, self.grid_num))
        A_aug = pd.concat(repeat(sensitive_features, self.grid_num))
        Y_values = pd.concat(repeat(y, self.grid_num))

        theta_list = [s for theta in self.grids for s in repeat(theta, self.n)]
        X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)

        X_aug.index = range(self.n * self.grid_num)
        A_aug.index = range(self.n * self.grid_num)
        Y_values.index = range(self.n * self.grid_num)

        W = (self.loss.eval([self._next_grid(_theta) for _theta in X_aug["theta"]], Y_values) -
                self.loss.eval([self._prev_grid(_theta) for _theta in X_aug["theta"]], Y_values))
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
            return 1 * ((pred_binary - X['theta']) >= 0)
        return self.utility_parity.gamma(classifier)

    def signed_weights(self, lambda_vec):
        return self.utility_parity.signed_weights(lambda_vec)

    def optimal_label(self, signed_weights):
        """ A heuristic is used to calculate the optimal y, as suggested in the
        section 4.4 of 'Agarwal et al. (2019) <https://arxiv.org/abs/1905.12843v1>'.

        Parameters
        ----------
        signed_weights : pandas.DataFrame
            signed weights
        
        Returns
        -------
        pandas.Series
            The optimal score y for each x before augmentation.
        """
        redY = 1 * (signed_weights > 0)
        redW = signed_weights.abs()
        redW = self.n * self.grid_num * redW / redW.sum()
        minimum = list()
        for i in range(self.n):
            index_set = [i + j * self.n for j in range(self.grid_num)]
            W_i = redW.iloc[index_set]
            Y_i = redY.iloc[index_set]
            Y_i.index = range(len(Y_i))
            W_i.index = range(len(Y_i))
            cost_i = {}
            for pred in self.pred_grid:
                cost_i[pred] = abs(Y_i - self.pred_vec[pred]).dot(W_i)
            minimum[i] = min(cost_i, key=cost_i.get)
        return pd.Series(minimum)
