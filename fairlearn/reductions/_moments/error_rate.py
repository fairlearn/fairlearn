# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import narwhals.stable.v1 as nw
import numpy as np

from fairlearn.utils._input_validation import _validate_and_reformat_input

from .moment import _ALL, _LABEL, ClassificationMoment

_MESSAGE_BAD_COSTS = (
    "costs needs to be a dictionary with keys "
    "'fp' and 'fn' containing non-negative values, which are not both zero"
)


@dataclass(frozen=True)
class ErrorRateResult:
    """Result returned by :meth:`ErrorRate.gamma`."""

    error: float


class ErrorRate(ClassificationMoment):
    r"""Misclassification error as a moment.

    A classifier :math:`h(X)` has the misclassification error equal to

    .. math::
      P[h(X) \ne Y]

    It is also possible to specify costs for false positives and false
    negatives. The error then evaluates to

    .. math::
      c_{FP} P[h(X)=1, Y=0] + c_{FN} P[h(X)=0, Y=1]

    where :math:`c_{FP}` and :math:`c_{FN}` are the costs of false positive
    and false negative errors respectively. The default misclassification
    error corresponds to :math:`c_{FP}=c_{FN}=1.0`.

    Read more in the :ref:`User Guide <error_rate>`.

    Parameters
    ----------
    costs : dict
        Dictionary with keys :code:`'fp'` and :code:`'fn'` containing the
        costs of false positives and false negatives. If none are provided
        costs of 1.0 are assumed.
    """

    def __init__(self, *, costs: dict[Literal["fp", "fn"], float] | None = None):
        """Initialize the costs."""
        super().__init__()
        if costs is None:
            self.fp_cost = 1.0
            self.fn_cost = 1.0
        elif (
            isinstance(costs, dict)
            and costs.keys() == {"fp", "fn"}
            and costs["fp"] >= 0.0
            and costs["fn"] >= 0.0
            and costs["fp"] + costs["fn"] > 0.0
        ):
            self.fp_cost = costs["fp"]
            self.fn_cost = costs["fn"]
        else:
            raise ValueError(_MESSAGE_BAD_COSTS)

    def load_data(self, X, y, *, sensitive_features) -> None:
        """Load the specified data into the object.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The feature array
        y : array-like of shape (n_samples,)
            The label vector
        sensitive_features : array-like of shape (n_samples, n_sensitive_features)
            The sensitive feature vector
        """
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
        )
        # TODO: remove following line when _validate_and_reformat_input returns sf_train
        # as a narwhals series (while with PR #1533 it comes as whatever native
        # namespace it was passed in):
        sf_train = nw.from_native(sf_train, pass_through=True, eager_only=True)
        # The following uses X so that the estimators get X untouched
        super().load_data(X, y_train, sensitive_features=sf_train)
        self._index = [_ALL]

    @property
    def index(self) -> list[str]:
        """Return the index listing the constraints."""
        return self._index

    def gamma(self, predictor: Callable) -> ErrorRateResult:
        """Calculate the weighted error for a predictor.

        Parameters
        ----------
        predictor : func
            bound function returning predictions

        Returns
        -------
        ErrorRateResult
            The weighted error is available through the :attr:`error` attribute.
        """
        # self.X passed into the predict function of an estimator needs not to be a
        # narwhals type, in case third party libraries don't depend on narwhals:
        y_pred = predictor(self.X)
        y_pred = nw.from_native(y_pred, pass_through=True, eager_only=True)
        if isinstance(y_pred, np.ndarray):
            # TensorFlow is returning an (n,1) array, which results
            # in the subtraction in the 'error =' line generating an
            # (n,n) array
            y_pred = np.squeeze(y_pred)
        signed_errors = self.tags[_LABEL] - y_pred
        total_fn_cost = (signed_errors.filter(signed_errors > 0) * self.fn_cost).sum()
        total_fp_cost = (signed_errors.filter(signed_errors < 0) * self.fp_cost * -1).sum()
        result = ErrorRateResult(
            error=float((total_fn_cost + total_fp_cost) / self.total_samples)
        )
        self._gamma_descr = str(result)
        return result

    def objective_value(self, predictor: Callable) -> float:
        """Calculate the scalar error objective for the predictor."""
        return self.gamma(predictor).error

    def project_lambda(self, lambda_vec: nw.typing.IntoSeries) -> nw.typing.IntoSeries:
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(
        self, lambda_vec: nw.typing.IntoSeries | None = None
    ) -> nw.typing.IntoSeries:
        """Return the signed weights."""
        weights = -self.fp_cost + (self.fp_cost + self.fn_cost) * self.tags[_LABEL]
        if lambda_vec is None:
            return weights.to_native()
        else:
            return lambda_vec[_ALL] * weights.to_native()
