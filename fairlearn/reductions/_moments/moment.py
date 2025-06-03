# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from collections.abc import Callable

import narwhals.stable.v1 as nw
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

    Our implementations of the reductions approach to fairness
    :footcite:p:`agarwal2018reductions` make use
    of :class:`Moment` objects to describe both the optimization objective
    and the fairness constraints
    imposed on the solution. This is an abstract class for all such objects.

    Read more in the :ref:`User Guide <reductions>`.
    """

    def __init__(self):
        self.data_loaded = False

    def load_data(
        self,
        X: nw.typing.IntoDataFrame,
        y: nw.typing.IntoSeries,
        *,
        sensitive_features: nw.typing.IntoSeries | None = None,
    ) -> None:
        """Load a set of data for use by this object.

        Parameters
        ----------
        X : numpy.ndarray, DataFrame object supported by narwhals, or list of lists
            The feature array.
        y : numpy.ndarray, Series object supported by narwhals or list
            The label vector.
        sensitive_features : numpy.ndarray, Series object supported by narwhals, or list, default=None
            The sensitive feature vector.
        """
        assert self.data_loaded is False, "data can be loaded only once"
        self.X = X
        self._y = y
        # TODO (when dependency from pandas is being removed): Dynamically change
        # backends for y, sensitive_features and self.tags to user's backend:
        y = nw.new_series(name="y", values=y, native_namespace=pd)
        sensitive_features = nw.new_series(
            name="sensitive_features", values=sensitive_features, native_namespace=pd
        )
        self.tags = nw.from_dict({_LABEL: y}, backend=pd)
        self.tags = self.tags.with_columns(**{_GROUP_ID: sensitive_features})
        self.data_loaded = True
        self._gamma_descr = None

    @property
    def total_samples(self) -> int:
        """Return the number of samples in the data."""
        return self.X.shape[0]

    @property
    def _y_as_series(self) -> pd.Series:
        """Return the y array as a :class:`~pandas.Series`."""
        return self._y

    @property
    def index(self) -> pd.MultiIndex | pd.Index:
        """Return a pandas (multi-)index listing the constraints."""
        raise NotImplementedError()

    def gamma(self, predictor: Callable) -> pd.Series:  # noqa: D102
        """Calculate the degree to which constraints are currently violated by the predictor."""
        raise NotImplementedError()

    def bound(self) -> pd.Series:  # noqa: D102
        """Return vector of fairness bound constraint the length of gamma."""
        raise NotImplementedError()

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:  # noqa: D102
        """Return the projected lambda values."""
        raise NotImplementedError()

    def signed_weights(self, lambda_vec: pd.Series) -> pd.Series:  # noqa: D102
        """Return the signed weights."""
        raise NotImplementedError()

    def _moment_type(self) -> type[Moment]:
        """Return the moment type, e.g., ClassificationMoment vs LossMoment."""
        return NotImplementedError()

    def default_objective(self) -> Moment:
        """Return the default objective for the moment."""
        raise NotImplementedError()


# Ensure that Moment shows up in correct place in documentation
# when it is used as a base class
Moment.__module__ = "fairlearn.reductions"


class ClassificationMoment(Moment):
    """Moment that can be expressed as weighted classification error."""

    def _moment_type(self):
        return ClassificationMoment


# Ensure that ClassificationMoment shows up in correct place in documentation
# when it is used as a base class
ClassificationMoment.__module__ = "fairlearn.reductions"


class LossMoment(Moment):
    """Moment that can be expressed as weighted loss."""

    def __init__(self, loss):
        super().__init__()
        self.reduction_loss = loss

    def _moment_type(self):
        return LossMoment


# Ensure that LossMoment shows up in correct place in documentation
# when it is used as a base class
LossMoment.__module__ = "fairlearn.reductions"
