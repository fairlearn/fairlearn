# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Callable, Literal, List

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd

from fairlearn.utils._input_validation import _validate_and_reformat_input

from .moment import _ALL, _LABEL, ClassificationMoment

_MESSAGE_BAD_COSTS = (
    "costs needs to be a dictionary with keys "
    "'fp' and 'fn' containing non-negative values, which are not both zero"
)


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
    and false negative errors respectively. The standard misclassification
    error corresponds to :math:`c_{FP}=c_{FN}=1.0`.

    Parameters
    ----------
    costs : dict
        The dictionary with keys :code:`'fp'` and :code:`'fn'` containing the
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

    def load_data(self, X, y, *, sensitive_features, control_features=None) -> None:
        """Load the specified data into the object."""
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            enforce_binary_labels=True,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        # TODO: remove following line when _validate_and_reformat_input is narwhalified
        # (because then sf_train comes as a narwhals series, while now it comes as a
        # pandas series):
        sf_train = nw.from_native(sf_train, pass_through=True, eager_only=True)
        # The following uses X so that the estimators get X untouched
        super().load_data(X, y_train, sensitive_features=sf_train)
        self._index = [_ALL]

    @property
    def index(self) -> List[str]:
        """Return the index listing the constraints."""
        return self._index

    def gamma(self, predictor: Callable) -> nw.typing.IntoSeries:
        """Return the gamma values for the given predictor."""
        pred = predictor(self.X)
        pred = nw.from_native(pred, pass_through=True, eager_only=True)
        if isinstance(pred, np.ndarray):
            # TensorFlow is returning an (n,1) array, which results
            # in the subtraction in the 'error =' line generating an
            # (n,n) array
            pred = np.squeeze(pred)
        signed_errors = self.tags[_LABEL] - pred
        total_fn_cost = (signed_errors.filter(signed_errors > 0) * self.fn_cost).sum()
        total_fp_cost = (signed_errors.filter(signed_errors < 0) * self.fp_cost * -1).sum()
        error_value = (total_fn_cost + total_fp_cost) / self.total_samples
        # TODO: at the moment, native_namespace only depends on the user-passed type
        # of X; but what should happen if the user pass inputs from several
        # dataframe libraries?
        if isinstance(self.X, np.ndarray):
            # TODO (when dependency from pandas is removed): remove this check to always
            # return the type that the user has passed; for now: if user has passed
            # np.array for X, still return a pd.Series as before
            error = nw.new_series(name="error", values=error_value, native_namespace=pd)
        else:
            error = nw.new_series(
                name="error",
                values=[error_value],
                native_namespace=nw.get_native_namespace(self.X),
            )
        self._gamma_descr = str(error)
        return error.to_native()

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
