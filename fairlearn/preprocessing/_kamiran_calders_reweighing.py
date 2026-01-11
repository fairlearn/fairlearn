from collections.abc import Iterable

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from fairlearn.utils._fixes import validate_data


class KamiranCaldersReweighing(TransformerMixin, BaseEstimator):
    r"""
    A reweighing transformer for mitigating bias with respect to sensitive features.

    ``KamiranCaldersReweighing`` assigns instance-level weights such that the
    distribution of the target variable becomes independent of one or more
    sensitive features. This implements the reweighing method proposed by
    Kamiran and Calders (2012).

    The transformer does **not** modify feature values. Instead, it augments the
    input data with a per-sample weight column, which can be passed to downstream
    estimators via a ``sample_weight`` argument during model training.

    The transformer supports NumPy arrays, pandas DataFrames, and Narwhals
    DataFrames as input. When DataFrame-like inputs are used, feature names
    are preserved and validated in a scikit-learn–compatible manner. Narwhals
    DataFrames are automatically converted for internal computation and returned
    as Narwhals DataFrames, preserving a roundtrip interface.

    Parameters
    ----------
    drop_target : bool, default=True
        Whether to drop the target column in the transformed output.
        Note that the current implementation does not modify the target column
        and always returns features augmented with a ``"weight"`` column.

    Notes
    -----
    Let :math:`S` denote a set of sensitive features and :math:`Y` the target
    variable. The goal of reweighing is to enforce statistical independence
    between :math:`S` and :math:`Y` in the weighted dataset.

    For each combination :math:`(s, y)` of sensitive feature values and target
    labels, the reweighing method computes a weight

    .. math::

        w(s, y) = \frac{P(S = s)\,P(Y = y)}{P(S = s, Y = y)}

    where probabilities are estimated empirically from the training data.

    Each instance with sensitive attributes :math:`s` and label :math:`y`
    is assigned the corresponding weight :math:`w(s, y)`. When training a
    model using these weights, the resulting weighted empirical distribution
    satisfies

    .. math::

        P_w(S = s, Y = y) = P(S = s)\,P(Y = y),

    thereby removing dependence between the sensitive features and the target.

    This method is most naturally applied to learning algorithms that support
    sample weighting.

    .. versionadded:: 0.14.0
    """

    def __init__(self, *, drop_target: bool = True):
        self.drop_target = drop_target

    def fit(self, X, y, *, sensitive_features=None, **fit_params):
        """
        Fit the reweighing transformer.

        Empirical probabilities involving sensitive features and the target
        are estimated and stored for computing instance-level weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features (NumPy, pandas, or Narwhals DataFrame).

        y : array-like of shape (n_samples,)
            Target labels.

        sensitive_features : iterable of int or str, optional
            Indices or names of sensitive feature columns. If None or empty,
            all samples receive unit weight.

        **fit_params : dict
            Ignored; for API compatibility.

        Returns
        -------
        self : KamiranCaldersReweighing
        """
        self._is_pandas_ = isinstance(X, pd.DataFrame)
        self._is_nw_ = isinstance(X, nw.DataFrame)

        X_valid = validate_data(self, X, ensure_2d=True, dtype=None)
        y_valid = validate_data(self, y, ensure_2d=False, dtype=None)
        n_samples = X_valid.shape[0]

        if self._is_pandas_:
            self.feature_names_in_ = X.columns.to_list()
        elif self._is_nw_:
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = None

        # Encode features
        if X_valid.dtype.kind in {"O", "U", "S"}:
            self._x_encoder_ = OrdinalEncoder()
            X_enc = self._x_encoder_.fit_transform(X_valid)
        else:
            self._x_encoder_ = None
            X_enc = X_valid

        # Encode target
        if y_valid.dtype.kind in {"O", "U", "S"}:
            self._y_encoder_ = OrdinalEncoder()
            y_enc = self._y_encoder_.fit_transform(y_valid.reshape(-1, 1)).ravel()
        else:
            self._y_encoder_ = None
            y_enc = y_valid
        self._y_array_ = y_enc

        # Handle no sensitive features
        if not sensitive_features:
            self.sensitive_idx_ = []
            self._weights_mapping_ = None
            return self

        # Validate sensitive features
        if isinstance(sensitive_features, Iterable):
            sensitive_features = list(sensitive_features)

        self.sensitive_idx_ = []
        for f in sensitive_features:
            if isinstance(f, int):
                if f < 0 or f >= X_valid.shape[1]:
                    raise ValueError(f"Sensitive feature index {f} is out of bounds")
                self.sensitive_idx_.append(f)
            else:
                if self.feature_names_in_ is None:
                    raise ValueError(f"Sensitive feature {f} provided, but X has no column names")
                if f not in self.feature_names_in_:
                    raise ValueError(f"Sensitive feature {f} not found in columns")
                self.sensitive_idx_.append(self.feature_names_in_.index(f))

        # Compute weights
        S = X_enc[:, self.sensitive_idx_]
        SY = np.column_stack([S, y_enc])

        unique_rows, counts = np.unique(SY, axis=0, return_counts=True)
        ps = np.array([(S == row[:-1]).all(axis=1).mean() for row in unique_rows])
        py = np.array([(y_enc == row[-1]).mean() for row in unique_rows])
        psy = counts / n_samples

        self._weights_mapping_ = {
            tuple(row): (p_s * p_y) / p_sy for row, p_s, p_y, p_sy in zip(unique_rows, ps, py, psy)
        }

        return self

    def transform(self, X, y=None):
        """
        Append instance-level weights to X.

        If sensitive features were provided at fit, and y is given, each
        sample receives the corresponding weight. Otherwise, all samples
        receive weight 1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features. NumPy, pandas, or Narwhals DataFrame.

        y : array-like of shape (n_samples,), optional
            Target labels for computing non-unit weights.

        Returns
        -------
        X_transformed : same type as input
            Input augmented with a ``weight`` column.
        """
        check_is_fitted(self)

        if isinstance(X, nw.DataFrame):
            X = X.to_pandas()  # ensure sklearn sees valid feature names

        X_valid = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)
        n_samples = X_valid.shape[0]

        X_enc = self._x_encoder_.transform(X_valid) if self._x_encoder_ else X_valid

        if self._weights_mapping_ is None or y is None:
            weights = np.ones(n_samples)
        else:
            y_array = np.asarray(y).ravel()
            if self._y_encoder_:
                y_array = self._y_encoder_.transform(y_array.reshape(-1, 1)).ravel()
            S = X_enc[:, self.sensitive_idx_]
            structured = np.column_stack([S, y_array])
            weights = np.array([self._weights_mapping_.get(tuple(row), 1.0) for row in structured])

        if self._is_pandas_:
            df_out = pd.DataFrame(X_valid, columns=self.feature_names_in_)
            df_out["weight"] = weights
            return df_out
        elif self._is_nw_:
            return nw.from_numpy(
                np.column_stack([X_valid, weights]),
                schema=list(self.feature_names_in_) + ["weight"],
                backend="pandas",
            )
        else:
            return np.column_stack([X_valid, weights])

    def fit_transform(self, X, y, *, sensitive_features=None, **fit_params):
        """
        Fit and transform X, returning it with a ``weight`` column.

        Equivalent to calling ``fit`` followed by ``transform``.
        """
        return self.fit(X, y, sensitive_features=sensitive_features).transform(X, y)
