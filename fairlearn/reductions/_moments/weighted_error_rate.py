import pandas as pd
import numpy as np
from itertools import repeat
from .moment import LossMoment,ClassificationMoment,_ALL, _LABEL
from .bounded_group_loss import SquareLoss
_WEIGHTS = "weights"

class WeightedErrorRate(LossMoment):
    """Weighted Loss error."""
    ## Not Sure about whether this should inherit from ClassificationMoment or LossMoment

    short_name = "Weighted Error Rate"
    def __init__(self,loss):
        super(WeightedErrorRate,self).__init__(loss)

    ## for what we need here is augmented data. Hence to avoid unnecessary calculation, we use augmented data having been calculated in regression_moment here
    ## and directly return in the function load_data to suit the inferface of _lagrangian.
    def load_augmented_data(self, X, y,**kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y, **kwargs)
        self.index = [_ALL] #“all”
        self.n = y.shape[0]
        if _WEIGHTS in kwargs:
            self.weights = kwargs[_WEIGHTS]
            self.weights = self.n * self.weights / self.weights.sum()
        else:
            self.weights = 1      
        self.tags[_WEIGHTS] = self.weights

    def load_data(self, X, y,**kwargs):
        return

    def gamma(self, predictor):
        def h(X): 
            return 1*( predictor(X.drop(['theta'], axis=1)) - X['theta'] >= 0)
        """Return the gamma values for the given predictor."""
        pred = h(self.X)
        error = pd.Series(data=(self.tags[_WEIGHTS]*(self.tags[_LABEL] - pred).abs()).mean(),
                          index=self.index)#here self.tags = y is the true label.
        self._gamma_descr = str(error)
        return error

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        if lambda_vec is None:
            return self.tags[_WEIGHTS] * (2 * self.tags[_LABEL] - 1)
        else:
            return lambda_vec[_ALL] * self.tags[_WEIGHTS] * (2 * self.tags[_LABEL] - 1)
