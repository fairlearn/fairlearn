# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_ALL = "all"
_SIGN = "sign"


class Moment:
    """Generic moment.

    Our implementations of the reductions approach to fairness described
    in `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_ make use
    of :class:`Moment` objects to describe the disparity constraints
    imposed on the solution. This is an abstract class for all such objects.
    """

    def __init__(self):
        self.data_loaded = False

    def load_data(self,
                  X,
                  y: pd.Series,
                  *,
                  sensitive_features: pd.Series = None):
        """Load a set of data for use by this object.

        Parameters
        ----------
        X : array
            The feature array
        y : pandas.Series
            The label vector
        sensitive_features : pandas.Series
            The sensitive feature vector (default None)
        """
        assert self.data_loaded is False, \
            "data can be loaded only once"
        if sensitive_features is not None:
            assert isinstance(sensitive_features, pd.Series)
        self.X = X
        self._y = y
        self.tags = pd.DataFrame({_LABEL: y})
        if sensitive_features is not None:
            self.tags[_GROUP_ID] = sensitive_features
        self.data_loaded = True
        self._gamma_descr = None

    @property
    def total_samples(self):
        """Return the number of samples in the data."""
        return self.X.shape[0]

    @property
    def _y_as_series(self):
        """Return the y array as a :class:`~pandas.Series`."""
        return self._y

    def gamma(self, predictor):  # noqa: D102
        """Calculate the degree to which constraints are currently violated by the predictor."""
        raise NotImplementedError()

    def bound(self):  # noqa: D102
        """Return vector of fairness bound constraint the length of gamma."""
        raise NotImplementedError()

    def project_lambda(self, lambda_vec):  # noqa: D102
        """Return the projected lambda values."""
        raise NotImplementedError()

    def signed_weights(self, lambda_vec):  # noqa: D102
        """Return the signed weights."""
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
