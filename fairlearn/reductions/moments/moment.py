# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_EVENT = "event"
_ALL = "all"


class Moment:
    """Generic moment"""

    def __init__(self):
        self.data_loaded = False

    def load_data(self, X, y, **kwargs):
        assert self.data_loaded is False, \
            "data can be loaded only once"
        self.X = X
        self.tags = pd.DataFrame({_LABEL: y})
        self.n = self.X.shape[0]
        self.data_loaded = True
        self._gamma_descr = None

    def gamma(self, predictor):
        raise NotImplementedError()

    def project_lambda(self, lambda_vec):
        raise NotImplementedError()

    def signed_weights(self, lambda_vec):
        raise NotImplementedError()


class ClassificationMoment(Moment):
    """Moment that can be expressed as weighted classification error"""


class LossMoment(Moment):
    """Moment that can be expressed as weighted loss"""

    def __init__(self, loss):
        super().__init__()
        self.reduction_loss = loss
