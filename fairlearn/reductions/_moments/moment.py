# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from fairlearn._input_validation import _KW_SENSITIVE_FEATURES

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_ALL = "all"
_SIGN = "sign"


class Moment:
    """Generic moment."""

    def __init__(self):
        self.data_loaded = False

    def load_data(self, X, y, **kwargs):
        """Load a set of data for use by this object.

        The keyword arguments can contain a :code:`sensitive_features` array.

        :param X: The feature data
        :type X: array

        :param y: The true label data
        :type y: array
        """
        assert self.data_loaded is False, \
            "data can be loaded only once"
        self.X = X
        self.n = self.X.shape[0]
        self.tags = pd.DataFrame({_LABEL: y})
        if _KW_SENSITIVE_FEATURES in kwargs:
            self.tags[_GROUP_ID] = kwargs[_KW_SENSITIVE_FEATURES]
        self.data_loaded = True
        self._gamma_descr = None

    def gamma(self, predictor):  # noqa: D102
        raise NotImplementedError()

    def project_lambda(self, lambda_vec):  # noqa: D102
        raise NotImplementedError()

    def signed_weights(self, lambda_vec):  # noqa: D102
        raise NotImplementedError()


# Ensure that Moment shows up in correct place in documentation
# when it is used as a base class
Moment.__module__ = "fairlearn.reductions"


class ClassificationMoment(Moment):
    """Moment that can be expressed as weighted classification error."""


# Ensure that ClassificationMoment shows up in correct place in documentation
# when it is used as a base class
ClassificationMoment.__module__ = "fairlearn.reductions"


class LossMoment(Moment):
    """Moment that can be expressed as weighted loss."""

    def __init__(self, loss):
        super().__init__()
        self.reduction_loss = loss


# Ensure that LossMoment shows up in correct place in documentation
# when it is used as a base class
LossMoment.__module__ = "fairlearn.reductions"
