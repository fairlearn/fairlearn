# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

_REDUCTION_TYPE_CLASSIFICATION = "classification"
_REDUCTION_TYPE_LOSS_MINIMIZATION = "loss_minimization"

class Moment:
    """Generic moment"""

    def __init__(self):
        self.initialized = False

    def init(self, dataX, dataA, dataY):
        assert self.initialized is False, \
            "moments can be initialized only once"
        self.X = dataX
        self.tags = pd.DataFrame(
            {"protected_attribute": dataA, "label": dataY})
        self.n = dataX.shape[0]
        self.initialized = True
        self._gamma_descr = None
