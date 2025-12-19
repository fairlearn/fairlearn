# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from collections.abc import Iterable

import narwhals.stable.v1 as nw
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class KamiranCaldersReweighing(TransformerMixin, BaseEstimator):
    r"""
    A reweighing transformer for mitigating bias with respect to sensitive features.

    KamiranCaldersReweighing assigns instance-level weights such that the distribution
    of the target variable becomes independent of one or more sensitive features.
    This implements the reweighing method proposed by Kamiran and Calders (2012).

    The resulting weights can be supplied to downstream estimators via a
    ``sample_weight`` argument during model training.

    Read more in the :ref:`User Guide <preprocessing>`.

    Parameters
    ----------
    drop_target : bool, default=True
        Whether to remove the target column from the transformed output.
        If False, the target column is retained alongside the computed weights.

    Notes
    -----
    Let :math:`S` denote a set of sensitive features and :math:`Y` the target
    variable. The goal of reweighing is to enforce statistical independence
    between :math:`S` and :math:`Y` in the weighted dataset.

    For each combination :math:`(s, y)` of sensitive feature values and target
    labels, the reweighing method computes a weight

    .. math::

        w(s, y) = \frac{P(S = s) \, P(Y = y)}{P(S = s, Y = y)}

    where probabilities are estimated empirically from the training data.

    Each instance with sensitive attributes :math:`s` and label :math:`y`
    is assigned the corresponding weight :math:`w(s, y)`. When training a
    model using these weights, the resulting weighted empirical distribution
    satisfies

    .. math::

        P_w(S = s, Y = y) = P(S = s) P(Y = y),

    thereby removing dependence between the sensitive features and the target.

    This method does not modify feature values; it only adjusts instance weights.
    As such, it is most naturally applied to learning algorithms that support
    sample weighting.

    .. versionadded:: 0.6
    """

    def __init__(self, *, drop_target: bool = True):
        self.drop_target = drop_target

    def _ensure_dataframe(self, X, *, schema=None, backend="pandas"):

        if isinstance(X, nw.DataFrame):
            return X

        if isinstance(X, np.ndarray):

            # Structured NumPy array
            if X.dtype.names is not None:
                return nw.from_numpy(X, backend=backend)

            # Plain 2-D NumPy array
            if X.ndim == 2:
                if schema is None:
                    schema = [f"x{i}" for i in range(X.shape[1])]
                return nw.from_numpy(X, schema=schema, backend=backend)

            raise TypeError("Unsupported NumPy array shape")

        return nw.from_native(X, pass_through=True, eager_only=True)

    def _ensure_series_as_frame(self, y, *, name="y", backend="pandas"):

        if isinstance(y, nw.DataFrame):
            if len(y.columns) != 1:
                raise ValueError("y must have exactly one column")
            return y

        if isinstance(y, np.ndarray):

            if y.ndim == 1:
                y = y.reshape(-1, 1)

            if y.ndim == 2 and y.shape[1] == 1:
                return nw.from_numpy(
                    y,
                    schema=[name],
                    backend=backend,
                )

            raise ValueError("y must be a 1-D array or a 2-D array with exactly one column")

        obj = nw.from_native(y, allow_series=True, eager_only=True)

        if isinstance(obj, nw.Series):
            df = obj.to_frame()
            return df.rename({df.columns[0]: name})

        if isinstance(obj, nw.DataFrame):
            if len(obj.columns) != 1:
                raise ValueError("y must have exactly one column")
            return obj

        raise TypeError("Unsupported type for y")

    def fit(self, X, y, sensitive_features: Iterable):
        r"""
        Learn reweighing factors from the training data.

        This method estimates the joint and marginal distributions of the
        sensitive features and the target variable, and computes reweighing
        factors under the assumption of independence.

        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix.
        y : Series, DataFrame, or array-like
            Target variable.
        sensitive_features : Iterable
            Columns in ``X`` corresponding to sensitive attributes.

        Returns
        -------
        self : KamiranCaldersReweighing
            Fitted transformer with learned reweighing factors.
        """
        X = self._ensure_dataframe(X)
        y = self._ensure_series_as_frame(y)

        # Handle sensitive_features for NumPy inputs
        if isinstance(sensitive_features, Iterable):
            sensitive_features = list(sensitive_features)

        # If sensitive_features are integers → positional columns
        if all(isinstance(f, int) for f in sensitive_features):
            try:
                sensitive_features = [X.columns[f] for f in sensitive_features]
            except IndexError:
                raise ValueError("Sensitive feature index out of bounds")

        # Validate column names
        missing_features = [f for f in sensitive_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"Sensitive feature(s) not found in X: {missing_features}")

        self.sensitive_features_ = list(sensitive_features)
        self.target_col_ = y.columns[0]
        self._y_ = y

        # Merge features and target for group operations
        merged = nw.concat([X, y], how="horizontal")
        total_rows = len(merged)
        feature_cols = self.sensitive_features_ + [self.target_col_]

        # Observed counts per combination of sensitive features and target
        observed_counts = merged.group_by(*feature_cols).agg(
            observed=nw.col(self.target_col_).count()
        )

        # Marginal counts
        marginal_s = merged.group_by(*self.sensitive_features_).agg(
            n_s=nw.col(self.target_col_).count()
        )
        marginal_y = merged.group_by(self.target_col_).agg(n_y=nw.col(self.target_col_).count())

        # Expected counts under independence assumption
        expected_counts = marginal_s.join(marginal_y, how="cross").with_columns(
            expected=nw.col("n_s") * nw.col("n_y") / total_rows
        )

        # Compute weights
        weights_df = observed_counts.join(
            expected_counts, on=feature_cols, how="left"
        ).with_columns(weight=nw.col("expected") / nw.col("observed"))
        self.weights_ = weights_df.drop(*["observed", "n_s", "n_y", "expected"])
        return self

    def transform(self, X, y=None):
        """
        Attach instance weights to a dataset.

        Each sample is assigned a weight based on its sensitive feature values
        and target label, as learned during ``fit``. These weights can be used
        directly in downstream estimators that support sample weighting.

        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix.
        y : Series, DataFrame, or array-like, optional
            Target variable. If not provided, the target passed during ``fit``
            is reused.

        Returns
        -------
        DataFrame
            Transformed dataset containing the original features and a
            ``weight`` column. The target column is included or excluded
            depending on the value of ``drop_target``.
        """
        check_is_fitted(self, ["weights_", "sensitive_features_", "target_col_"])

        X = self._ensure_dataframe(X)
        y = self._y_ if y is None else self._ensure_series_as_frame(y)

        merged = nw.concat([X, y], how="horizontal")
        merged = merged.join(
            self.weights_, on=self.sensitive_features_ + [self.target_col_], how="left"
        )

        return merged if not self.drop_target else merged.drop([self.target_col_])

    def fit_transform(self, X, y, sensitive_features: Iterable, **fit_params):
        """
        Fit the reweighing transformer and return the weighted dataset.

        Parameters
        ----------
        X : DataFrame or array-like
            Feature matrix.
        y : Series, DataFrame, or array-like
            Target variable.
        sensitive_features : Iterable
            Columns in ``X`` corresponding to sensitive attributes.

        Returns
        -------
        DataFrame
            Dataset augmented with instance-level reweighing factors.
        """
        return self.fit(X, y, sensitive_features).transform(X, y)
