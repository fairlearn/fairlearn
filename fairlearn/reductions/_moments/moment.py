# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from collections.abc import Callable

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

    def load_data(self, X, y: pd.Series, *, sensitive_features: pd.Series | None = None) -> None:
        """Load a set of data for use by this object.

        Parameters
        ----------
        X : array
            The feature array
        y : :class:`pandas.Series`
            The label vector
        sensitive_features : :class:`pandas.Series`
            The sensitive feature vector (default None)
        """
        if self.data_loaded:
            raise ValueError("data can be loaded only once")
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

    def gamma(self, predictor: Callable) -> pd.Series:
        """Calculate the degree to which constraints are currently violated by the predictor.

        Parameters
        ----------
        predictor : Callable
            A function that maps the feature matrix to predictions.

        Returns
        -------
        pandas.Series
            The value of each constraint for the predictor, indexed by :attr:`index`.
            The constraints are satisfied when these values do not exceed the values
            returned by :meth:`bound`.
        """
        raise NotImplementedError()

    def bound(self) -> pd.Series:
        """Return vector of fairness bound constraint the length of gamma.

        Returns
        -------
        pandas.Series
            The upper bound for each constraint, indexed by :attr:`index`. The
            entries of :meth:`gamma` are compared against these values.
        """
        raise NotImplementedError()

    def project_lambda(self, lambda_vec: pd.Series) -> pd.Series:
        """Return the projected lambda values.

        Parameters
        ----------
        lambda_vec : pandas.Series
            The vector of Lagrange multipliers, indexed by :attr:`index`.

        Returns
        -------
        pandas.Series
            Lagrange multipliers that are equivalent to `lambda_vec` for the
            Lagrangian but can have a smaller norm. They are used by the
            :class:`~fairlearn.reductions.ExponentiatedGradient` algorithm when
            evaluating the Lagrangian.
        """
        raise NotImplementedError()

    def signed_weights(self, lambda_vec: pd.Series) -> pd.Series:
        """Return the signed weights.

        Parameters
        ----------
        lambda_vec : pandas.Series
            The vector of Lagrange multipliers, indexed by :attr:`index`.

        Returns
        -------
        pandas.Series
            One weight per sample. The reductions algorithms add the weights of the
            objective and the constraints, use the sign of the sum to choose the label
            for a cost-sensitive learning problem, and the absolute value as the
            sample weight when calling the underlying estimator.
        """
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
