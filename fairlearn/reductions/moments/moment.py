# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

_REDUCTION_TYPE_CLASSIFICATION = "classification"
_REDUCTION_TYPE_LOSS_MINIMIZATION = "loss_minimization"

_GROUP_ID = "group_id"
_LABEL = "label"


class Moment:
    """Generic moment"""

    def __init__(self):
        self.initialized = False

    def init(self, dataX, dataA, dataY):
        assert self.initialized is False, \
            "moments can be initialized only once"
        self.X = dataX
        self.tags = pd.DataFrame(
            {_GROUP_ID: dataA, _LABEL: dataY})
        self.n = dataX.shape[0]
        self.initialized = True
        self._gamma_descr = None

    def gamma(self, predictor):
        raise NotImplementedError()

    def project_lambda(self, lambda_vec):
        raise NotImplementedError()

    def signed_weights(self, lambda_vec):
        raise NotImplementedError()
