# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Iterable

import narwhals.stable.v1 as nw
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


def _mid_rank_percentiles(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the percentile rank of each value within a reference sample.

    Values are ranked with the mid-rank convention, so a value that occurs
    several times in the reference sample gets the average of the ranks its
    copies span. This keeps ties tied and puts every rank strictly inside
    :math:`(0, 1)`, which means the corresponding quantile is always defined.
    """
    ordered = np.sort(reference)
    below = np.searchsorted(ordered, values, side="left")
    at_or_below = np.searchsorted(ordered, values, side="right")
    return (below + 0.5 * (at_or_below - below)) / ordered.shape[0]


class PercentileRepairer(TransformerMixin, BaseEstimator):
    r"""Repair a column so that its distribution is the same in every group.

    ``PercentileRepairer`` implements the percentile repair procedure of
    Feldman et al. (2015). Every value keeps its percentile rank inside its
    own group, but the rank is read back off the *median* distribution of all
    groups instead of the group's own distribution. Once every group has been
    mapped that way, each of them follows the same median distribution, so the
    repaired column no longer carries information about the sensitive feature.

    Read more in the :ref:`User Guide <percentile_repairer>`.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    sensitive_feature_ids : list, default=()
        Column that defines the groups, as a single element. This can be an
        int, in the case of numpy, or a string, in the case of pandas. The
        column is passed through unchanged; every other column is repaired.
        Exactly one sensitive feature is supported.

    repair_level : float, default=1.0
        How much of the repair to apply, between 0 and 1. With
        ``repair_level=1`` a value is replaced by its repaired counterpart,
        with ``repair_level=0`` the column is returned unchanged, and values
        in between are the linear interpolation of the two.

    Attributes
    ----------
    groups_ : numpy.ndarray
        The distinct values of the sensitive feature seen during ``fit``, in
        ascending order.

    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    Let :math:`S` be the sensitive feature and let :math:`g` index its groups.
    Write :math:`F_g` for the empirical cumulative distribution function of a
    repaired column inside group :math:`g`, and :math:`Q_g` for its quantile
    function. The *median distribution* of the column is the quantile function

    .. math::

        \bar{Q}(p) = \operatorname{median}_g Q_g(p) \quad \text{for } p \in [0, 1],

    that is, the pointwise median over the groups of their quantile functions.
    A value :math:`v` in group :math:`g` is repaired to

    .. math::

        v' = (1 - \lambda) v + \lambda \bar{Q}(F_g(v)),

    where :math:`\lambda` is ``repair_level``. Because :math:`F_g` is
    non-decreasing, the repair is monotone inside each group: two values that
    are ordered stay ordered, and two values that are equal stay equal. At
    :math:`\lambda = 1` the repaired values of every group follow the median
    distribution.

    Only ordered values are repaired. Categorical information has no meaningful
    percentiles, so every column, including the sensitive one, must be numeric:
    encode categorical columns first, for example with
    :func:`pandas.get_dummies`. Columns with missing values are rejected, since
    a missing value has no percentile.

    A group that was not seen during ``fit`` has no percentile mapping, so
    :meth:`transform` raises a :class:`ValueError` for it. Values outside the
    range seen during ``fit`` are mapped to the ends of the median
    distribution.

    A target column is repaired the same way as any other column: pass it as
    one of the columns of ``X``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fairlearn.preprocessing import PercentileRepairer
    >>> data = pd.DataFrame(
    ...     {"group": [0, 0, 0, 1, 1, 1], "score": [1000, 1100, 1200, 1300, 1400, 1500]}
    ... )
    >>> repaired = PercentileRepairer(sensitive_feature_ids=["group"]).fit_transform(data)
    >>> [round(float(score), 1) for score in repaired[:, 1]]
    [1150.0, 1250.0, 1350.0, 1150.0, 1250.0, 1350.0]
    >>> for group in (0, 1):
    ...     print(sorted(round(float(score), 1) for score in repaired[data["group"] == group, 1]))
    [1150.0, 1250.0, 1350.0]
    [1150.0, 1250.0, 1350.0]
    """

    def __init__(self, *, sensitive_feature_ids: Iterable = (), repair_level: float = 1.0):
        self.sensitive_feature_ids = sensitive_feature_ids
        self.repair_level = repair_level

    def _check_params(self) -> None:
        """Validate the parameters that do not depend on the data."""
        n_sensitive = len(list(self.sensitive_feature_ids))
        if n_sensitive != 1:
            raise ValueError(
                "sensitive_feature_ids must name exactly one sensitive feature, "
                f"got {n_sensitive}. Repairing against several sensitive features at "
                "once is not supported."
            )
        if not 0 <= self.repair_level <= 1:
            raise ValueError(f"repair_level must be between 0 and 1, got {self.repair_level}.")

    def _prepare(self, X, *, reset: bool) -> tuple[np.ndarray, int]:
        """Return X as a 2d float array together with the sensitive column index."""
        sensitive_feature_ids = list(self.sensitive_feature_ids)
        X_nw = nw.from_native(X, pass_through=True, eager_only=True)
        columns = None
        if isinstance(X_nw, nw.DataFrame):
            columns = list(X_nw.columns)
            if reset:
                self._columns_in_ = columns
            elif self._columns_in_ is not None and columns != self._columns_in_:
                raise ValueError(
                    "Named columns must match the columns and order seen during fit. "
                    f"Expected columns: {self._columns_in_}; got: {columns}."
                )
            values = self._to_float(X_nw.to_numpy(), reset=reset)
            missing_columns = [c for c in sensitive_feature_ids if c not in columns]
        else:
            if reset:
                self._columns_in_ = None
            values = self._to_float(X, reset=reset)
            missing_columns = [
                i
                for i in sensitive_feature_ids
                if not (isinstance(i, (int, np.integer)) and 0 <= i < values.shape[1])
            ]

        if len(missing_columns) > 0:
            raise ValueError(f"Columns {missing_columns} not found in the input data.")

        if columns is not None:
            return values, columns.index(sensitive_feature_ids[0])
        return values, sensitive_feature_ids[0]

    def _to_float(self, values, *, reset: bool) -> np.ndarray:
        """Validate the input and return it as a 2d array of floats."""
        try:
            return validate_data(self, values, dtype=np.float64, reset=reset)
        except ValueError as error:
            # scikit-learn rejects a non-numeric column with a bare conversion
            # error; point at the encoding step instead.
            if "could not convert" in str(error) or "invalid literal" in str(error):
                raise ValueError(
                    "PercentileRepairer can only repair numeric columns: every column of X, "
                    "including the sensitive feature, must hold numbers. Encode categorical "
                    "columns first, for example with pandas.get_dummies or with an ordinal "
                    "encoding."
                ) from error
            raise

    def fit(self, X, y=None):
        """Learn the percentile mapping of every group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The dataset, holding both the sensitive feature and the columns to
            repair. pandas DataFrame, Polars DataFrame, and PyArrow Table
            inputs are supported.
        y : Ignored
            Not used; present for API compatibility with scikit-learn.

        Returns
        -------
        self : object
            Fitted PercentileRepairer instance.

        Raises
        ------
        ValueError
            If ``sensitive_feature_ids`` does not name exactly one column of
            ``X``, if ``repair_level`` is outside ``[0, 1]``, or if a column
            holds missing or non-numeric values.
        """
        self._check_params()
        values, sensitive = self._prepare(X, reset=True)

        sensitive_values = values[:, sensitive]
        self.groups_ = np.unique(sensitive_values)
        self._sensitive_column_ = sensitive
        self._repaired_columns_ = [i for i in range(values.shape[1]) if i != sensitive]

        # The distribution of each repaired column inside each group, learnt once
        # and reused by transform.
        self._group_values_ = {
            group: values[sensitive_values == group][:, self._repaired_columns_]
            for group in self.groups_
        }

        return self

    def transform(self, X):
        """Repair every non-sensitive column of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The dataset to repair. It must have the same columns, in the same
            order, as the ``X`` passed to ``fit``, and every row must belong to
            a group seen during ``fit``.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            The repaired dataset. The sensitive column is unchanged, and every
            other column is replaced by its repaired value.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If a row belongs to a group that was not seen during ``fit``, or if
            the columns of ``X`` do not match the ones seen during ``fit``.
        """
        check_is_fitted(self, ["groups_", "_sensitive_column_"])

        values, sensitive = self._prepare(X, reset=False)
        sensitive_values = values[:, sensitive]

        known = np.isin(sensitive_values, self.groups_)
        if not np.all(known):
            unseen = np.unique(sensitive_values[~known])
            raise ValueError(
                f"The sensitive feature contains group(s) {unseen.tolist()} that were not "
                "seen during fit. PercentileRepairer can only map groups it has a "
                "distribution for."
            )

        repaired_values = values.copy()
        for position, column in enumerate(self._repaired_columns_):
            references = [self._group_values_[group][:, position] for group in self.groups_]
            for group, reference in zip(self.groups_, references, strict=True):
                rows = sensitive_values == group
                percentiles = _mid_rank_percentiles(values[rows, column], reference)
                group_quantiles = np.stack(
                    [np.quantile(ref, percentiles, method="inverted_cdf") for ref in references]
                )
                median_quantiles = np.median(group_quantiles, axis=0)
                repaired_values[rows, column] = (1 - self.repair_level) * values[
                    rows, column
                ] + self.repair_level * median_quantiles

        return repaired_values

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Every column of the input is kept, so the output names are the input
        names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If ``None``, the column names seen during
            ``fit`` are used, or ``["x0", "x1", ...]`` if ``X`` had no column
            names. If given, it must match the column names seen during
            ``fit``.

        Returns
        -------
        numpy.ndarray of str
            The names of the transformed features.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If ``input_features`` does not match the features seen during
            ``fit``.
        """
        check_is_fitted(self)

        if input_features is None:
            if self._columns_in_ is not None:
                input_features = list(self._columns_in_)
            else:
                input_features = [f"x{i}" for i in range(self.n_features_in_)]
        elif len(input_features) != self.n_features_in_:
            raise ValueError(
                f"input_features should have length equal to number of features "
                f"({self.n_features_in_}), got {len(input_features)}"
            )
        elif self._columns_in_ is not None and list(input_features) != self._columns_in_:
            raise ValueError("input_features is not equal to the column names seen during fit")

        return np.asarray(input_features, dtype=object)
