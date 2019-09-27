# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from .moment import Moment


class MisclassificationError(Moment):
    """Misclassification error"""
    short_name = "Err"

    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataY, dataY)
        self.index = ["all"]

    def gamma(self, predictor):
        pred = predictor(self.X)
        error = pd.Series(data=(self.tags["label"] - pred).abs().mean(),
                          index=self.index)
        self._gamma_descr = str(error)
        return error

    def lambda_signed(self, lambda_vec):
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        if lambda_vec is None:
            return 2 * self.tags["label"] - 1
        else:
            return lambda_vec["all"] * (2 * self.tags["label"] - 1)
